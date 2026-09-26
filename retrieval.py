"""Retrieval core shared by the API, the ingest script, the eval and the tests.

Everything that decides what a query returns lives here, so the eval measures
exactly the code path the API serves.
"""
import re

import pandas as pd
from chromadb.utils import embedding_functions

from preprocess import preprocess_text

COLLECTION = "Description_Vector"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Default cut-off on cosine similarity for /similar_products (None = no floor).
DEFAULT_MIN_SIMILARITY = None


def embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


def price_to_int(price) -> int:
    """Prices arrive as '$999'. Store a number so Chroma can filter on it."""
    digits = re.sub(r"[^\d]", "", str(price))
    if not digits:
        raise ValueError(f"unparseable price: {price!r}")
    return int(digits)


def create_collection(client, name: str = COLLECTION, reset: bool = False):
    """Get or create the cosine-space collection, optionally dropping it first."""
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return client.get_or_create_collection(
        name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function(),
    )


def index_products(collection, frame: pd.DataFrame, preprocess=preprocess_text) -> int:
    """Embed each description; name, price, category and rating ride along as metadata.

    price_usd and Rating are numeric so the API can filter on them inside the query.
    """
    collection.add(
        ids=[str(i) for i in range(len(frame))],
        documents=[preprocess(d) for d in frame["Description"]],
        metadatas=[
            {
                "Product Name": row["Product Name"],
                "Description": row["Description"],
                "Price": row["Price"],
                "price_usd": price_to_int(row["Price"]),
                "Category": row["Category"],
                "Rating": float(row["Rating"]),
            }
            for _, row in frame.iterrows()
        ],
    )
    return collection.count()


def build_where(min_price: int, max_price: int, min_rating: float, max_rating: float) -> dict:
    """Price and rating bounds as a Chroma `where` clause (all bounds inclusive)."""
    return {
        "$and": [
            {"price_usd": {"$gte": min_price}},
            {"price_usd": {"$lte": max_price}},
            {"Rating": {"$gte": min_rating}},
            {"Rating": {"$lte": max_rating}},
        ]
    }


def search(
    collection,
    query: str,
    *,
    min_price: int = 0,
    max_price: int = 10_000,
    min_rating: float = 0.0,
    max_rating: float = 5.0,
    limit: int = 3,
    min_similarity: float | None = None,
    preprocess=preprocess_text,
) -> list[dict]:
    """Filter and retrieve in one step.

    The nearest neighbours are chosen from products that already satisfy the
    filters, so a product in range is never lost for sitting outside a fixed
    candidate window. Results below `min_similarity` are dropped.
    """
    found = collection.query(
        query_texts=[preprocess(query)],
        n_results=limit,
        where=build_where(min_price, max_price, min_rating, max_rating),
        include=["metadatas", "distances"],
    )

    results = []
    for metadata, distance in zip(found["metadatas"][0], found["distances"][0]):
        similarity = round(1 - distance, 3)  # the collection uses cosine distance
        if min_similarity is not None and similarity < min_similarity:
            continue
        item = {k: v for k, v in metadata.items() if k != "price_usd"}
        item["similarity"] = similarity
        results.append(item)
    return results
