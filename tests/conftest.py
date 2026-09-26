import os
from pathlib import Path

import pytest

# app.py refuses to start without a token; set one before it is imported.
os.environ["API_TOKEN"] = "test-token"

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def catalogue_collection():
    """The real catalogue indexed into an in-process Chroma (downloads the model once)."""
    import chromadb
    import pandas as pd
    from chromadb.config import Settings

    import retrieval

    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False, allow_reset=True))
    collection = retrieval.create_collection(client, "test_catalogue", reset=True)
    retrieval.index_products(collection, pd.read_csv(ROOT / "data" / "Generated_Product_Data.csv"))
    return collection
