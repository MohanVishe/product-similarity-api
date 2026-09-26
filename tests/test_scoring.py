import pytest

from evals.scoring import (
    above_floor,
    first_relevant_rank,
    parser_stats,
    pick_floor,
    recall_at_k,
    reciprocal_rank,
    separation,
    summarize,
)

RANKED = ["Watch", "Laptop", "Backpack", "Smartphone"]


def test_first_relevant_rank():
    assert first_relevant_rank(RANKED, ["Backpack"]) == 3
    assert first_relevant_rank(RANKED, ["Dress"]) is None


def test_recall_at_k():
    assert recall_at_k(RANKED, ["Laptop", "Smartphone"], 1) == 0.0
    assert recall_at_k(RANKED, ["Laptop", "Smartphone"], 3) == 0.5
    assert recall_at_k(RANKED, ["Laptop", "Smartphone"], 4) == 1.0
    with pytest.raises(ValueError):
        recall_at_k(RANKED, [], 3)


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, ["Laptop"]) == 0.5
    assert reciprocal_rank(RANKED, ["Dress"]) == 0.0


def test_above_floor_keeps_scores_at_or_over_the_floor():
    ranking = [["A", 0.5], ["B", 0.25], ["C", 0.1]]
    assert above_floor(ranking, None) == ["A", "B", "C"]
    assert above_floor(ranking, 0.25) == ["A", "B"]


RECORDS = [
    # relevant at rank 1, and at rank 3
    {"type": "paraphrase", "relevant": ["A"], "ranking": [["A", 0.6], ["B", 0.3], ["C", 0.2]]},
    {"type": "intent", "relevant": ["C"], "ranking": [["A", 0.5], ["B", 0.4], ["C", 0.3]]},
    # negation whose ruled-out product comes first
    {"type": "negation", "relevant": ["B"], "excluded": ["A"],
     "ranking": [["A", 0.5], ["B", 0.45], ["C", 0.1]]},
    {"type": "off_topic", "relevant": [], "ranking": [["A", 0.15], ["B", 0.1]]},
    {"type": "off_topic", "relevant": [], "ranking": [["C", 0.05]]},
]


def test_summarize_known_answer_without_floor():
    s = summarize(RECORDS, None)
    assert s["n_on_topic"] == 3 and s["n_off_topic"] == 2
    assert s["recall@1"] == round(1 / 3, 4)
    assert s["recall@3"] == 1.0
    assert s["mrr"] == round((1 + 1 / 3 + 1 / 2) / 3, 4)
    assert s["off_topic_fp_rate"] == 1.0
    assert s["on_topic_empty_rate"] == 0.0
    assert s["negation_excluded_at_1"] == 1.0
    assert s["negation_excluded_in_top3"] == 1.0
    assert s["negation_accuracy"] == 0.0


def test_summarize_known_answer_with_floor():
    s = summarize(RECORDS, 0.35)
    # the intent query keeps only A and B (C at 0.3 is dropped), so its relevant item is gone
    assert s["recall@3"] == round(2 / 3, 4)
    assert s["mrr"] == round((1 + 0 + 1 / 2) / 3, 4)
    assert s["off_topic_fp_rate"] == 0.0
    assert s["on_topic_empty_rate"] == 0.0


def test_separation():
    assert separation(RECORDS) == {
        "off_topic_best_similarity_max": 0.15,
        "on_topic_first_relevant_similarity_min": 0.3,
        "on_topic_relevant_in_top3_similarity_min": 0.3,
    }


def test_pick_floor_prefers_lowest_fp_within_recall_budget():
    sweep = [
        {"min_similarity": None, "recall@3": 0.9, "off_topic_fp_rate": 1.0},
        {"min_similarity": 0.1, "recall@3": 0.9, "off_topic_fp_rate": 0.5},
        {"min_similarity": 0.2, "recall@3": 0.86, "off_topic_fp_rate": 0.0},
        {"min_similarity": 0.3, "recall@3": 0.86, "off_topic_fp_rate": 0.0},
        {"min_similarity": 0.4, "recall@3": 0.5, "off_topic_fp_rate": 0.0},
    ]
    assert pick_floor(sweep, max_recall_loss=0.05) == 0.2
    assert pick_floor(sweep, max_recall_loss=0.0) == 0.1


def test_pick_floor_returns_none_when_no_floor_helps():
    sweep = [
        {"min_similarity": None, "recall@3": 0.9, "off_topic_fp_rate": 1.0},
        {"min_similarity": 0.1, "recall@3": 0.9, "off_topic_fp_rate": 1.0},
    ]
    assert pick_floor(sweep) is None


def test_rankings_may_carry_a_mode_score():
    ranking = [["A", 0.5, 0.0328], ["B", 0.2, 0.0164]]
    assert above_floor(ranking, 0.3) == ["A"]  # the floor is on the cosine (field 1)
    records = [
        {"type": "exact", "relevant": ["B"], "ranking": ranking},
        {"type": "off_topic", "relevant": [], "ranking": [["C", 0.1, 0.0161]]},
    ]
    assert separation(records, index=2) == {
        "off_topic_best_similarity_max": 0.0161,
        "on_topic_first_relevant_similarity_min": 0.0164,
        "on_topic_relevant_in_top3_similarity_min": 0.0164,
    }


def test_parser_stats():
    records = [
        {"type": "negation", "relevant": ["B"], "excluded": ["A"], "parser_excluded": ["A"]},
        {"type": "negation", "relevant": ["B"], "excluded": ["C"], "parser_excluded": []},
        {"type": "hedge", "relevant": ["W"], "excluded": [], "parser_excluded": ["W"]},
        {"type": "exact", "relevant": ["X"], "parser_excluded": []},
    ]
    assert parser_stats(records) == {
        "n_negation_with_excluded": 2,
        "negations_fully_excluded": 1,
        "relevant_wrongly_excluded": 1,
        "n_hedge": 1,
        "hedges_with_exclusion": 1,
    }
