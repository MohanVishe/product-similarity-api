"""Product similarity API.

Semantic search over product descriptions (Chroma + all-MiniLM-L6-v2),
with price and rating filters applied inside the vector query.

    uvicorn app:app --reload --port 8080
"""
import os

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, model_validator

from preprocess import preprocess_text

load_dotenv()

COLLECTION = "Description_Vector"
EMBED_MODEL = "all-MiniLM-L6-v2"

API_TOKEN = os.getenv("API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("API_TOKEN is not set. Copy .env.example to .env and set one.")

client = chromadb.HttpClient(
    host=os.getenv("CHROMA_HOST", "localhost"),
    port=int(os.getenv("CHROMA_PORT", "8000")),
)
collection = client.get_collection(
    COLLECTION,
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    ),
)

app = FastAPI(
    title="Product Similarity API",
    description="Semantic product search with price and rating filters.",
    version="1.0.0",
)


class SimilarProductsRequest(BaseModel):
    name: str = Field(..., description="What to search for, in natural language")
    min_price: int = Field(0, ge=0)
    max_price: int = Field(10_000, ge=0)
    min_rating: float = Field(0.0, ge=0, le=5)
    max_rating: float = Field(5.0, ge=0, le=5)
    limit: int = Field(3, ge=1, le=20)
    min_similarity: float | None = Field(
        None, ge=-1, le=1,
        description="Optional cut-off on cosine similarity; results below it are dropped",
    )

    @model_validator(mode="after")
    def _bounds_in_order(self):
        if self.min_price > self.max_price:
            raise ValueError("min_price must not exceed max_price")
        if self.min_rating > self.max_rating:
            raise ValueError("min_rating must not exceed max_rating")
        return self


async def authenticate(token: str | None = Header(None)):
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
async def health():
    return {"status": "ok", "collection": COLLECTION, "documents": collection.count()}


@app.post("/similar_products", dependencies=[Depends(authenticate)])
async def similar_products(request: SimilarProductsRequest):
    """Filter and retrieve in one step.

    Price and rating go into Chroma's `where` clause, so the nearest neighbours
    are chosen from products that already satisfy the filters. A product in
    range is never lost for sitting outside a fixed candidate window.
    """
    where = {
        "$and": [
            {"price_usd": {"$gte": request.min_price}},
            {"price_usd": {"$lte": request.max_price}},
            {"Rating": {"$gte": request.min_rating}},
            {"Rating": {"$lte": request.max_rating}},
        ]
    }
    found = collection.query(
        query_texts=[preprocess_text(request.name)],
        n_results=request.limit,
        where=where,
        include=["metadatas", "distances"],
    )

    results = []
    for metadata, distance in zip(found["metadatas"][0], found["distances"][0]):
        similarity = round(1 - distance, 3)  # the collection uses cosine distance
        if request.min_similarity is not None and similarity < request.min_similarity:
            continue
        item = {k: v for k, v in metadata.items() if k != "price_usd"}
        item["similarity"] = similarity
        results.append(item)

    return {"query": request.name, "count": len(results), "results": results}
