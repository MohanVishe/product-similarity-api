"""Product similarity API.

Semantic search over product descriptions (Chroma + all-MiniLM-L6-v2),
with price and rating filters applied to the retrieved candidates.

    uvicorn app:app --reload --port 8080
"""
import os
import re

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

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


async def authenticate(token: str | None = Header(None)):
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _price_to_int(price) -> int | None:
    """Prices arrive as '$999'. Strip anything that isn't a digit."""
    digits = re.sub(r"[^\d]", "", str(price))
    return int(digits) if digits else None


@app.get("/health")
async def health():
    return {"status": "ok", "collection": COLLECTION, "documents": collection.count()}


@app.post("/similar_products", dependencies=[Depends(authenticate)])
async def similar_products(request: SimilarProductsRequest):
    """Retrieve semantically, then filter.

    Note the ordering: Chroma returns the nearest neighbours by description
    similarity, and price/rating filters are applied to that candidate set.
    A very cheap product that is semantically distant will not surface -
    see the limitations section of the README.
    """
    candidates = collection.query(
        query_texts=[preprocess_text(request.name)],
        n_results=max(request.limit * 5, 20),
    )

    matches = []
    for metadata in candidates["metadatas"][0]:
        price = _price_to_int(metadata.get("Price"))
        rating = metadata.get("Rating")

        if price is None or rating is None:
            continue
        if not (request.min_price <= price <= request.max_price):
            continue
        if not (request.min_rating <= rating <= request.max_rating):
            continue

        matches.append(metadata)
        if len(matches) == request.limit:
            break

    return {"query": request.name, "count": len(matches), "results": matches}
