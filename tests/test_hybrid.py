"""BM25, reciprocal rank fusion and the ranking modes."""
import pytest

import retrieval


def test_rrf_known_answer():
    fused = retrieval.reciprocal_rank_fusion([["a", "b", "c"], ["b", "a"]], k=60)
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)  # missing from the second ranking: no credit


def test_rrf_rewards_agreement_over_one_strong_list():
    fused = retrieval.reciprocal_rank_fusion([["x", "y"], ["y", "z"]])
    assert max(fused, key=fused.get) == "y"


def names(results):
    return [r["Product Name"] for r in results]


def test_bm25_matches_brand_words(catalogue_collection):
    # "XYZ" appears only in the laptop's description.
    assert names(retrieval.search(catalogue_collection, "XYZ", mode="bm25")) == ["Laptop"]
    assert names(retrieval.search(catalogue_collection, "XYZ", mode="hybrid"))[0] == "Laptop"


def test_bm25_returns_nothing_without_a_shared_word(catalogue_collection):
    assert retrieval.search(catalogue_collection, "quantum chromodynamics", mode="bm25") == []


def test_bm25_also_indexes_name_and_category(catalogue_collection):
    # "footwear" is only the Running Shoes category, not in any description.
    assert names(retrieval.search(catalogue_collection, "footwear", mode="bm25")) == ["Running Shoes"]


def test_hybrid_ranks_every_candidate_and_keeps_cosine(catalogue_collection):
    results = retrieval.search(catalogue_collection, "bluetooth", mode="hybrid", limit=16)
    assert len(results) == 16  # the dense side ranks every product
    assert set(names(results[:2])) == {"Wireless Headphones", "Portable Speaker"}
    assert all(-1 <= r["similarity"] <= 1 for r in results)
    # a product found by both lists outscores one found by the dense list alone
    assert results[0]["score"] > 1 / 61 >= results[-1]["score"]


def test_modes_respect_filters(catalogue_collection):
    for mode in retrieval.MODES:
        results = retrieval.search(catalogue_collection, "bluetooth", mode=mode, max_price=70, limit=16)
        assert results and "Wireless Headphones" not in names(results)  # $79


@pytest.mark.parametrize("mode", retrieval.MODES)
def test_excluded_products_never_come_back(catalogue_collection, mode):
    found = retrieval.search_detailed(catalogue_collection, "not a laptop", mode=mode, limit=16)
    assert found["excluded"] == ["Laptop"]
    assert "Laptop" not in names(found["results"])


def test_exclusions_can_be_switched_off(catalogue_collection):
    found = retrieval.search_detailed(catalogue_collection, "not a laptop", exclusions=False)
    assert found["excluded"] == [] and names(found["results"])[0] == "Laptop"


def test_default_mode_is_a_known_mode():
    assert retrieval.DEFAULT_MODE in retrieval.MODES
