"""Pure scoring functions for the retrieval eval (no Chroma, no model).

A "ranking" is the list of (product name, similarity) pairs the search returned,
best first. A "record" is one labelled query plus its ranking.
"""
from statistics import mean

ON_TOPIC_TYPES = ("paraphrase", "intent", "negation", "filtered")


def above_floor(ranking, floor):
    """Names the API would return at this min_similarity (None = no floor)."""
    if floor is None:
        return [name for name, _ in ranking]
    return [name for name, sim in ranking if sim >= floor]


def first_relevant_rank(ranked, relevant):
    """1-based rank of the first relevant name, or None if none was returned."""
    relevant = set(relevant)
    for position, name in enumerate(ranked, start=1):
        if name in relevant:
            return position
    return None


def recall_at_k(ranked, relevant, k):
    """Share of the relevant products that appear in the top k."""
    if not relevant:
        raise ValueError("recall is undefined for a query with no relevant products")
    return len(set(ranked[:k]) & set(relevant)) / len(set(relevant))


def reciprocal_rank(ranked, relevant):
    rank = first_relevant_rank(ranked, relevant)
    return 0.0 if rank is None else 1.0 / rank


def _mean(values):
    return round(mean(values), 4) if values else None


def summarize(records, floor, k_values=(1, 3, 5), limit=3):
    """Aggregate metrics for one min_similarity setting.

    - recall@k and MRR over queries that have relevant products (on-topic).
    - off_topic_fp_rate: share of off-topic queries for which the API would
      return at least one product (its top result clears the floor).
    - on_topic_empty_rate: share of on-topic queries that return nothing.
    - negation_excluded_at_1 / in_top_limit: share of negation queries whose
      ruled-out product is ranked first / appears in the first `limit` results.
    """
    on_topic = [r for r in records if r["relevant"]]
    off_topic = [r for r in records if r["type"] == "off_topic"]
    negations = [r for r in records if r.get("excluded")]

    def ranked(r):
        return above_floor(r["ranking"], floor)

    summary = {"min_similarity": floor, "n_on_topic": len(on_topic), "n_off_topic": len(off_topic)}
    for k in k_values:
        summary[f"recall@{k}"] = _mean([recall_at_k(ranked(r), r["relevant"], k) for r in on_topic])
    summary["mrr"] = _mean([reciprocal_rank(ranked(r), r["relevant"]) for r in on_topic])
    summary["on_topic_empty_rate"] = _mean([float(not ranked(r)[:limit]) for r in on_topic])
    summary["off_topic_fp_rate"] = _mean([float(bool(ranked(r)[:limit])) for r in off_topic])
    summary["n_negation_with_excluded"] = len(negations)
    summary["negation_excluded_at_1"] = _mean(
        [float(ranked(r)[:1] != [] and ranked(r)[0] in r["excluded"]) for r in negations]
    )
    summary[f"negation_excluded_in_top{limit}"] = _mean(
        [float(bool(set(ranked(r)[:limit]) & set(r["excluded"]))) for r in negations]
    )
    return summary


def by_type(records, floor, k_values=(1, 3, 5)):
    """recall@k and MRR per on-topic query type."""
    out = {}
    for query_type in ON_TOPIC_TYPES:
        group = [r for r in records if r["type"] == query_type and r["relevant"]]
        if not group:
            continue
        row = {"n": len(group)}
        for k in k_values:
            row[f"recall@{k}"] = _mean(
                [recall_at_k(above_floor(r["ranking"], floor), r["relevant"], k) for r in group]
            )
        row["mrr"] = _mean([reciprocal_rank(above_floor(r["ranking"], floor), r["relevant"]) for r in group])
        out[query_type] = row
    return out


def separation(records, limit=3):
    """How far apart on-topic and off-topic similarities sit (no floor).

    A floor can only be clean if the best off-topic score is below the weakest
    correct hit; the gap between them is the room a threshold has.
    """
    off_topic_top = [r["ranking"][0][1] for r in records if r["type"] == "off_topic" and r["ranking"]]
    first_relevant, relevant_in_top = [], []
    for r in records:
        if not r["relevant"]:
            continue
        hits = [sim for name, sim in r["ranking"] if name in r["relevant"]]
        if hits:
            first_relevant.append(hits[0])
        relevant_in_top += [sim for name, sim in r["ranking"][:limit] if name in r["relevant"]]
    return {
        "off_topic_best_similarity_max": max(off_topic_top) if off_topic_top else None,
        "on_topic_first_relevant_similarity_min": min(first_relevant) if first_relevant else None,
        f"on_topic_relevant_in_top{limit}_similarity_min": min(relevant_in_top) if relevant_in_top else None,
    }


def pick_floor(sweep, max_recall_loss=0.05, k=3):
    """The rule for a default floor, fixed before looking at the sweep.

    Among floors that cost at most `max_recall_loss` of recall@k compared with
    no floor, take the one with the lowest off-topic false-positive rate; on a
    tie, the lowest floor. Returns None if no floor helps (FP rate unchanged).
    """
    baseline = next(row for row in sweep if row["min_similarity"] is None)
    eligible = [
        row for row in sweep
        if row["min_similarity"] is not None
        and row[f"recall@{k}"] >= baseline[f"recall@{k}"] - max_recall_loss - 1e-9
    ]
    best = min(eligible, key=lambda row: (row["off_topic_fp_rate"], row["min_similarity"]))
    if best["off_topic_fp_rate"] >= baseline["off_topic_fp_rate"]:
        return None
    return best["min_similarity"]
