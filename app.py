"""Product similarity API.

Semantic search over product descriptions (Chroma + all-MiniLM-L6-v2),
with price and rating filters applied inside the vector query.

    uvicorn app:app --reload --port 8080
"""
import os
from functools import lru_cache

import chromadb
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, model_validator

import retrieval

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("API_TOKEN is not set. Copy .env.example to .env and set one.")


@lru_cache(maxsize=1)
def get_collection():
    """The indexed collection on the Chroma server (overridden in tests)."""
    client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "localhost"),
        port=int(os.getenv("CHROMA_PORT", "8000")),
    )
    return client.get_collection(
        retrieval.COLLECTION, embedding_function=retrieval.embedding_function()
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
        retrieval.DEFAULT_MIN_SIMILARITY, ge=-1, le=1,
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
async def health(collection=Depends(get_collection)):
    return {"status": "ok", "collection": retrieval.COLLECTION, "documents": collection.count()}


@app.post("/similar_products", dependencies=[Depends(authenticate)])
async def similar_products(request: SimilarProductsRequest, collection=Depends(get_collection)):
    """Price and rating go into Chroma's `where` clause; see retrieval.search."""
    results = retrieval.search(
        collection,
        request.name,
        min_price=request.min_price,
        max_price=request.max_price,
        min_rating=request.min_rating,
        max_rating=request.max_rating,
        limit=request.limit,
        min_similarity=request.min_similarity,
    )
    return {"query": request.name, "count": len(results), "results": results}
