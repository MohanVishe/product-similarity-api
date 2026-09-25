"""Build the Chroma collection from the product CSV.

Run once before starting the API. Requires a Chroma server:

    chroma run --path ./chroma-data --port 8000
    python ingest.py
"""
import argparse
import os
import re

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from preprocess import preprocess_text

COLLECTION = "Description_Vector"
EMBED_MODEL = "all-MiniLM-L6-v2"


def price_to_int(price) -> int:
    """Prices arrive as '$999'. Store a number so Chroma can filter on it."""
    digits = re.sub(r"[^\d]", "", str(price))
    if not digits:
        raise ValueError(f"unparseable price: {price!r}")
    return int(digits)


def main():
    load_dotenv()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/Generated_Product_Data.csv")
    ap.add_argument("--reset", action="store_true", help="drop the collection first")
    args = ap.parse_args()

    frame = pd.read_csv(args.csv)
    print(f"{len(frame)} products from {args.csv}")

    client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "localhost"),
        port=int(os.getenv("CHROMA_PORT", "8000")),
    )

    if args.reset:
        try:
            client.delete_collection(COLLECTION)
            print(f"dropped existing collection {COLLECTION!r}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        COLLECTION,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        ),
    )

    # The description is what gets embedded - it carries the semantics.
    # Name, price, category and rating ride along as metadata; price_usd and
    # Rating are numeric so the API can filter on them inside the query.
    collection.add(
        ids=[str(i) for i in range(len(frame))],
        documents=[preprocess_text(d) for d in frame["Description"]],
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

    print(f"indexed {collection.count()} documents into {COLLECTION!r}")


if __name__ == "__main__":
    main()
