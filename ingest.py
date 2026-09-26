"""Build the Chroma collection from the product CSV.

Run once before starting the API. Requires a Chroma server:

    chroma run --path ./chroma-data --port 8000
    python ingest.py
"""
import argparse
import os

import chromadb
import pandas as pd
from dotenv import load_dotenv

import retrieval


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
        print(f"dropping collection {retrieval.COLLECTION!r} if it exists")
    collection = retrieval.create_collection(client, reset=args.reset)
    count = retrieval.index_products(collection, frame)
    print(f"indexed {count} documents into {retrieval.COLLECTION!r}")


if __name__ == "__main__":
    main()
