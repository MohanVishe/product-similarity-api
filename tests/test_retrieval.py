import pytest
from pydantic import ValidationError

import retrieval
from app import SimilarProductsRequest


def test_where_has_all_four_inclusive_bounds():
    assert retrieval.build_where(10, 200, 4.0, 4.8) == {
        "$and": [
            {"price_usd": {"$gte": 10}},
            {"price_usd": {"$lte": 200}},
            {"Rating": {"$gte": 4.0}},
            {"Rating": {"$lte": 4.8}},
        ]
    }


def test_where_filters_on_numeric_fields_only():
    where = retrieval.build_where(0, 10_000, 0.0, 5.0)
    fields = {next(iter(clause)) for clause in where["$and"]}
    assert fields == {"price_usd", "Rating"}


def test_price_to_int():
    assert retrieval.price_to_int("$999") == 999
    with pytest.raises(ValueError):
        retrieval.price_to_int("free")


@pytest.mark.parametrize(
    "bounds",
    [{"min_price": 500, "max_price": 100}, {"min_rating": 4.9, "max_rating": 4.0}],
)
def test_request_rejects_inverted_bounds(bounds):
    with pytest.raises(ValidationError):
        SimilarProductsRequest(name="laptop", **bounds)


def test_request_defaults():
    request = SimilarProductsRequest(name="laptop")
    assert (request.min_price, request.max_price, request.limit) == (0, 10_000, 3)
    assert request.min_similarity == retrieval.DEFAULT_MIN_SIMILARITY
    assert request.mode == retrieval.DEFAULT_MODE and request.exclusions is True


def test_request_rejects_unknown_mode():
    with pytest.raises(ValidationError):
        SimilarProductsRequest(name="laptop", mode="sparse")



class _FakeCollection:
    """Records the query the API sends; returns canned neighbours."""

    id = "fake"
    _metadatas = [
        {"Product Name": "A", "Category": "Toys", "Description": "a red toy", "price_usd": 10, "Rating": 4.0},
        {"Product Name": "B", "Category": "Toys", "Description": "a blue toy", "price_usd": 20, "Rating": 4.5},
    ]

    def __init__(self):
        self.calls = []

    def count(self):
        return len(self._metadatas)

    def get(self, include):
        return {"ids": ["0", "1"], "metadatas": self._metadatas}

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return {"ids": [["0", "1"]], "metadatas": [self._metadatas], "distances": [[0.2, 0.7]]}


def test_search_passes_filters_into_the_query_and_applies_the_floor():
    fake = _FakeCollection()
    results = retrieval.search(fake, "Not a laptop", max_price=50, limit=2, min_similarity=0.5)
    (call,) = fake.calls
    # "laptop" names no product in this catalogue: nothing is excluded, the text is kept.
    assert call["where"] == retrieval.build_where(0, 50, 0.0, 5.0)
    assert call["query_texts"] == ["not laptop"]
    assert call["n_results"] == 2
    # similarity = 1 - cosine distance; B (0.3) is below the 0.5 floor.
    assert [(r["Product Name"], r["similarity"], r["score"]) for r in results] == [("A", 0.8, 0.8)]
    assert "price_usd" not in results[0]


def test_where_adds_excluded_names():
    where = retrieval.build_where(0, 100, 0.0, 5.0, excluded=["Laptop"])
    assert where["$and"][-1] == {"Product Name": {"$nin": ["Laptop"]}}


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        retrieval.search(_FakeCollection(), "toy", mode="sparse")
