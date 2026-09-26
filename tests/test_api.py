"""In-process API tests: FastAPI TestClient against an ephemeral Chroma."""
import pytest
from fastapi.testclient import TestClient

import app as app_module

AUTH = {"token": "test-token"}


@pytest.fixture()
def client_with():
    def make(collection):
        app_module.app.dependency_overrides[app_module.get_collection] = lambda: collection
        return TestClient(app_module.app)

    yield make
    app_module.app.dependency_overrides.clear()


class _Untouchable:
    def query(self, **kwargs):
        raise AssertionError("the vector store must not be queried for an invalid request")


def test_inverted_price_bounds_return_422_without_querying(client_with):
    client = client_with(_Untouchable())
    response = client.post(
        "/similar_products", headers=AUTH,
        json={"name": "laptop", "min_price": 500, "max_price": 100},
    )
    assert response.status_code == 422
    assert "min_price must not exceed max_price" in response.text


def test_missing_token_returns_401(client_with):
    client = client_with(_Untouchable())
    assert client.post("/similar_products", json={"name": "laptop"}).status_code == 401


def test_health_counts_the_catalogue(client_with, catalogue_collection):
    body = client_with(catalogue_collection).get("/health").json()
    assert body["status"] == "ok" and body["documents"] == 16


def test_search_end_to_end_respects_filters(client_with, catalogue_collection):
    client = client_with(catalogue_collection)
    response = client.post("/similar_products", headers=AUTH, json={
        "name": "wireless headphones for music", "max_price": 70, "limit": 5,
    })
    assert response.status_code == 200
    results = response.json()["results"]
    assert results, "expected in-range products"
    assert all(int(r["Price"].lstrip("$")) <= 70 for r in results)
    assert "Wireless Headphones" not in {r["Product Name"] for r in results}  # $79, filtered out
    assert results[0]["Product Name"] == "Portable Speaker"
    assert "price_usd" not in results[0] and 0 < results[0]["similarity"] <= 1


def test_min_similarity_drops_weak_matches(client_with, catalogue_collection):
    client = client_with(catalogue_collection)
    body = client.post("/similar_products", headers=AUTH, json={
        "name": "quantum chromodynamics lecture notes", "min_similarity": 0.3,
    }).json()
    assert body["count"] == 0 and body["results"] == []
