"""Retrieval core shared by the API, the ingest script, the eval and the tests.

Everything that decides what a query returns lives here, so the eval measures
exactly the code path the API serves.
"""
import re

import pandas as pd
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from exclusions import ParsedQuery, Product, parse_query
from preprocess import preprocess_text

COLLECTION = "Description_Vector"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Default cut-off on cosine similarity for /similar_products (None = no floor).
DEFAULT_MIN_SIMILARITY = None

# Ranking modes. "dense" is the original embedding search; "bm25" is keyword
# search over name + category + description; "hybrid" fuses the two rankings.
MODES = ("dense", "bm25", "hybrid")
# Picked by a rule fixed before the dev-set run (evals/run_eval.py DEFAULT_MODE_RULE):
# hybrid unless it lowers dev MRR or recall@3 below dense. It did (MRR 0.922 vs
# 0.938, both with exclusions), so dense stays the default; the eval fails if
# this constant and the rule disagree.
DEFAULT_MODE = "dense"
# Reciprocal rank fusion constant, the value from Cormack, Clarke & Buettcher
# (SIGIR 2009). Not tuned on the eval.
RRF_K = 60


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


def build_where(
    min_price: int, max_price: int, min_rating: float, max_rating: float, excluded=()
) -> dict:
    """Price and rating bounds (inclusive), plus any excluded product names, as a Chroma `where`."""
    clauses = [
        {"price_usd": {"$gte": min_price}},
        {"price_usd": {"$lte": max_price}},
        {"Rating": {"$gte": min_rating}},
        {"Rating": {"$lte": max_rating}},
    ]
    if excluded:
        clauses.append({"Product Name": {"$nin": list(excluded)}})
    return {"$and": clauses}


def bm25_tokens(text: str) -> list[str]:
    """Keyword tokens: the same normalisation as the embedded text."""
    return preprocess_text(text).split()


class Catalogue:
    """What the keyword side needs: product names/categories and a BM25 index.

    Built from the collection itself (names, categories and the original
    descriptions are stored as metadata), so the API, eval and tests need no
    second copy of the data. BM25 indexes name + category + description.
    """

    def __init__(self, ids, metadatas):
        self.ids = list(ids)
        self.products = [Product(m["Product Name"], m["Category"]) for m in metadatas]
        self.bm25 = BM25Okapi(
            [bm25_tokens(f"{m['Product Name']} {m['Category']} {m['Description']}") for m in metadatas]
        )

    def bm25_scores(self, text: str) -> dict:
        """BM25 score per document id (0 for documents sharing no word with the query)."""
        return dict(zip(self.ids, self.bm25.get_scores(bm25_tokens(text))))


_CATALOGUES: dict = {}


def catalogue_for(collection) -> Catalogue:
    """The Catalogue for a collection, rebuilt whenever its document count changes."""
    key = (str(getattr(collection, "id", id(collection))), collection.count())
    if key not in _CATALOGUES:
        got = collection.get(include=["metadatas"])
        _CATALOGUES.clear()
        _CATALOGUES[key] = Catalogue(got["ids"], got["metadatas"])
    return _CATALOGUES[key]


def reciprocal_rank_fusion(rankings, k: int = RRF_K) -> dict:
    """RRF: score(d) = sum over rankings of 1 / (k + rank of d), ranks from 1.

    A document missing from a ranking gets nothing from it. Rank-based, so the
    cosine and BM25 scales never have to be reconciled.
    """
    fused: dict = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return fused


def search_detailed(
    collection,
    query: str,
    *,
    min_price: int = 0,
    max_price: int = 10_000,
    min_rating: float = 0.0,
    max_rating: float = 5.0,
    limit: int = 3,
    min_similarity: float | None = None,
    mode: str = DEFAULT_MODE,
    exclusions: bool = True,
    preprocess=preprocess_text,
) -> dict:
    """Filter, exclude and rank in one step; returns results plus what the parser did.

    - Price, rating and excluded products go into Chroma's `where`, so the
      candidates are exactly the products that satisfy them.
    - dense: nearest neighbours by cosine similarity.
    - bm25: candidates that share at least one word with the query, by BM25.
    - hybrid: reciprocal rank fusion of the dense ranking (every candidate) and
      the BM25 ranking (candidates with a keyword match).
    Every result carries `similarity` (cosine, as before) and `score` (what the
    mode ranked by). `min_similarity` applies to the cosine in every mode.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    catalogue = catalogue_for(collection) if (exclusions or mode != "dense") else None
    parsed = parse_query(query, catalogue.products) if exclusions else ParsedQuery(query, query)

    # Dense mode needs only the top `limit`; the other modes rank every candidate.
    n_results = limit if mode == "dense" else max(limit, collection.count())
    found = collection.query(
        query_texts=[preprocess(parsed.text)],
        n_results=n_results,
        where=build_where(min_price, max_price, min_rating, max_rating, parsed.excluded),
        include=["metadatas", "distances"],
    )
    candidates = {
        doc_id: (metadata, round(1 - distance, 3))  # the collection uses cosine distance
        for doc_id, metadata, distance in zip(
            found["ids"][0], found["metadatas"][0], found["distances"][0]
        )
    }
    dense_order = list(candidates)  # nearest first

    if mode == "dense":
        ranked = [(doc_id, candidates[doc_id][1]) for doc_id in dense_order]
    else:
        bm25 = catalogue.bm25_scores(parsed.text)
        dense_rank = {doc_id: i for i, doc_id in enumerate(dense_order)}
        keyword_order = sorted(
            (doc_id for doc_id in dense_order if bm25.get(doc_id, 0.0) > 0),
            key=lambda doc_id: (-bm25[doc_id], dense_rank[doc_id]),  # ties: dense order
        )
        if mode == "bm25":
            ranked = [(doc_id, round(float(bm25[doc_id]), 4)) for doc_id in keyword_order]
        else:
            fused = reciprocal_rank_fusion([dense_order, keyword_order])
            order = sorted(dense_order, key=lambda doc_id: (-fused[doc_id], dense_rank[doc_id]))
            ranked = [(doc_id, round(fused[doc_id], 5)) for doc_id in order]

    results = []
    for doc_id, score in ranked:
        metadata, similarity = candidates[doc_id]
        if min_similarity is not None and similarity < min_similarity:
            continue
        item = {k: v for k, v in metadata.items() if k != "price_usd"}
        item["similarity"] = similarity
        item["score"] = score
        results.append(item)
        if len(results) == limit:
            break
    return {"results": results, "excluded": parsed.excluded, "query_text": parsed.text}


def search(collection, query: str, **kwargs) -> list[dict]:
    """The ranked results only; see `search_detailed` for the arguments."""
    return search_detailed(collection, query, **kwargs)["results"]
