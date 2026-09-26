"""Retrieval eval: labelled queries -> the API's own search code -> metrics.

Indexes the catalogue into an in-process (ephemeral) Chroma with the same
code ingest.py uses, runs every query through retrieval.search_detailed (what
/similar_products calls) in each ranking mode, with and without exclusion
parsing, and writes evals/results.json and evals/RESULTS.md. No Chroma server.

Two query sets:
- dev (queries.json, 40): its findings (negation ignored, no clean floor) are
  what the hybrid + exclusion changes were designed from, so it is in-sample
  for them.
- held-out (queries-heldout.json, 34): written and committed before the
  changes, scored once after them.

    python -m evals.run_eval                # regenerate results.json + RESULTS.md
    python -m evals.run_eval --check        # fail if numbers drift from the committed files
    python -m evals.run_eval --only dev     # print one set's summary, write nothing
"""
import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.config import Settings

import retrieval
from evals.scoring import (
    by_type,
    first_relevant_rank,
    parser_stats,
    pick_floor,
    separation,
    summarize,
)

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
CSV = ROOT / "data" / "Generated_Product_Data.csv"
RESULTS_JSON = HERE / "results.json"
RESULTS_MD = HERE / "RESULTS.md"

SETS = {
    "dev": {
        "file": "queries.json",
        "label": "Dev set (40 queries, in-sample for these changes)",
        "note": "The changes were designed from this set's findings (negation ignored, no clean "
                "floor), so after-numbers on it are in-sample.",
    },
    "heldout": {
        "file": "queries-heldout.json",
        "label": "Held-out set (34 queries, scored once after the changes)",
        "note": "Written from the product rows and committed (08efe32) before the hybrid and "
                "exclusion code existed; not used to design or tune anything. Same author as "
                "the parser, so not independent.",
    },
}

# (name, mode, exclusion parsing). "dense" with no exclusions is the API before this change.
CONFIGS = [
    ("dense", "dense", False),
    ("bm25", "bm25", False),
    ("hybrid", "hybrid", False),
    ("dense+excl", "dense", True),
    ("bm25+excl", "bm25", True),
    ("hybrid+excl", "hybrid", True),
]
BEFORE = "dense"
AFTER = f"{retrieval.DEFAULT_MODE}+excl"  # the API default after this change

K_VALUES = (1, 3, 5)
LIMIT = 3  # the API's default `limit`
FLOORS = [None] + [round(0.05 * i, 2) for i in range(0, 13)]  # none, 0.00 .. 0.60
COMPACT_FLOORS = [None, 0.15, 0.2, 0.25, 0.3]
MAX_RECALL_LOSS = 0.05
SIM_TOLERANCE = 0.002  # per-query similarities may wobble in the 3rd decimal across CPUs
DEFAULT_MODE_RULE = (
    "fixed before the dev run: hybrid is the default unless, with exclusions on, it lowers "
    "dev-set MRR or recall@3 below dense; otherwise dense. The held-out set is not consulted."
)


def _identity(text):
    return text


def run_queries(collection, queries, n_products, *, mode, exclusions, preprocess=retrieval.preprocess_text):
    records = []
    for q in queries:
        filters = q.get("filters", {})
        kwargs = dict(mode=mode, exclusions=exclusions, preprocess=preprocess, **filters)
        full = retrieval.search_detailed(collection, q["query"], limit=n_products, **kwargs)
        # The eval ranks the whole (filtered) catalogue so MRR is defined; check that
        # its top LIMIT is exactly what the API's default call returns.
        api_default = retrieval.search(collection, q["query"], limit=LIMIT, **kwargs)
        names = [r["Product Name"] for r in full["results"]]
        if [r["Product Name"] for r in api_default] != names[:LIMIT]:
            raise RuntimeError(f"{q['id']} ({mode}): limit={LIMIT} and full-ranking results disagree")
        records.append(
            {
                "id": q["id"],
                "type": q["type"],
                "query": q["query"],
                "filters": filters,
                "relevant": q["relevant"],
                "excluded": q.get("excluded", []),
                "parser_excluded": full["excluded"],
                "query_text": full["query_text"],
                "first_relevant_rank": first_relevant_rank(names, q["relevant"]) if q["relevant"] else None,
                "ranking": [[r["Product Name"], r["similarity"], r["score"]] for r in full["results"]],
            }
        )
    return records


def evaluate_set(set_name, collection, n_products, raw_collection=None):
    spec = SETS[set_name]
    queries = json.loads((HERE / spec["file"]).read_text(encoding="utf-8"))["queries"]
    counts = {}
    for q in queries:
        counts[q["type"]] = counts.get(q["type"], 0) + 1

    configs, per_config_records = {}, {}
    for name, mode, exclusions in CONFIGS:
        records = run_queries(collection, queries, n_products, mode=mode, exclusions=exclusions)
        per_config_records[name] = records
        configs[name] = {
            "mode": mode,
            "exclusions": exclusions,
            "no_floor": summarize(records, None, K_VALUES, LIMIT),
            "by_type": by_type(records, None, K_VALUES),
            "floors": [summarize(records, floor, K_VALUES, LIMIT) for floor in COMPACT_FLOORS],
            "separation_cosine": separation(records, LIMIT, index=1),
            "separation_score": separation(records, LIMIT, index=2),
            "parser": parser_stats(records) if exclusions else None,
        }

    after_records = per_config_records[AFTER]
    sweep = [summarize(after_records, floor, K_VALUES, LIMIT) for floor in FLOORS]
    queries_out = []
    for i, record in enumerate(after_records):
        record = dict(record)
        record["top3_by_config"] = {
            name: [entry[0] for entry in per_config_records[name][i]["ranking"][:LIMIT]]
            for name, _, _ in CONFIGS
        }
        record["first_relevant_rank_by_config"] = {
            name: per_config_records[name][i]["first_relevant_rank"] for name, _, _ in CONFIGS
        }
        queries_out.append(record)

    out = {
        "file": spec["file"],
        "label": spec["label"],
        "note": spec["note"],
        "n_queries": len(queries),
        "queries_by_type": counts,
        "configs": configs,
        "after_sweep": sweep,
        "floor_picked_by_rule": pick_floor(sweep, MAX_RECALL_LOSS, LIMIT),
    }
    if raw_collection is not None:
        raw = run_queries(raw_collection, queries, n_products, mode="dense", exclusions=False,
                          preprocess=_identity)
        out["ablation_raw_text"] = {
            "note": "Dense, no exclusions (the pre-change API), with descriptions and queries "
                    "embedded without preprocess_text.",
            "no_floor": summarize(raw, None, K_VALUES, LIMIT),
            "no_floor_by_type": by_type(raw, None, K_VALUES),
        }
    out["queries"] = queries_out
    return out


def default_mode_by_rule(dev):
    hybrid = dev["configs"]["hybrid+excl"]["no_floor"]
    dense = dev["configs"]["dense+excl"]["no_floor"]
    keep = hybrid["mrr"] >= dense["mrr"] and hybrid[f"recall@{LIMIT}"] >= dense[f"recall@{LIMIT}"]
    return "hybrid" if keep else "dense"


def evaluate(only=None):
    frame = pd.read_csv(CSV)
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False, allow_reset=True))
    coll = retrieval.create_collection(client, "eval_api_pipeline", reset=True)
    n_products = retrieval.index_products(coll, frame)

    results = {
        "about": "Generated by `python -m evals.run_eval`. Do not edit by hand; "
                 "`python -m evals.run_eval --check` fails if this drifts.",
        "config": {
            "embedding_model": retrieval.EMBED_MODEL,
            "distance": "cosine (similarity = 1 - distance, rounded to 3 dp as the API returns it)",
            "keyword": "BM25Okapi (rank-bm25) over name + category + description, same normalisation",
            "fusion": f"reciprocal rank fusion, k = {retrieval.RRF_K} (not tuned)",
            "n_products": n_products,
            "k_values": list(K_VALUES),
            "limit": LIMIT,
            "configs": {name: {"mode": mode, "exclusions": exc} for name, mode, exc in CONFIGS},
            "before": BEFORE,
            "after": AFTER,
            "api_default_mode": retrieval.DEFAULT_MODE,
            "api_default_min_similarity": retrieval.DEFAULT_MIN_SIMILARITY,
            "default_mode_rule": DEFAULT_MODE_RULE,
            "floor_rule": f"lowest off-topic FP rate among floors costing at most "
                          f"{MAX_RECALL_LOSS} recall@{LIMIT} vs no floor; ties -> lowest floor",
        },
    }
    for set_name in SETS:
        if only and set_name != only:
            continue
        raw_coll = None
        if set_name == "dev":
            raw_coll = retrieval.create_collection(client, "eval_raw_text", reset=True)
            retrieval.index_products(raw_coll, frame, preprocess=_identity)
        results[set_name] = evaluate_set(set_name, coll, n_products, raw_coll)
    if "dev" in results:
        results["config"]["default_mode_by_rule"] = default_mode_by_rule(results["dev"])
    return results


def _environment():
    return {
        pkg: version(pkg)
        for pkg in ("chromadb", "sentence-transformers", "torch", "transformers", "nltk", "rank-bm25")
    }


def _fmt(value):
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _fmt_floor(value):
    return "none" if value is None else f"{value:.2f}"


def _top(ranking, n=LIMIT, score=False):
    if not ranking:
        return "(nothing)"
    if score:
        return ", ".join(f"{e[0]} ({e[1]:.3f} / {e[2]:g})" for e in ranking[:n])
    return ", ".join(f"{e[0]} ({e[1]:.3f})" for e in ranking[:n])


def _config_table(data):
    lines = [
        "| config | recall@1 | recall@3 | recall@5 | MRR | ruled-out ranked 1st | negation accuracy | "
        "off-topic FP (no floor) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, cfg in data["configs"].items():
        s = cfg["no_floor"]
        lines.append(
            f"| {name} | {_fmt(s['recall@1'])} | {_fmt(s['recall@3'])} | {_fmt(s['recall@5'])} | "
            f"{_fmt(s['mrr'])} | {_fmt(s['negation_excluded_at_1'])} | {_fmt(s['negation_accuracy'])} | "
            f"{_fmt(s['off_topic_fp_rate'])} |"
        )
    return lines


def _floor_tables(data):
    header = "| config | " + " | ".join(_fmt_floor(f) for f in COMPACT_FLOORS) + " |"
    rule = "|---|" + "---|" * len(COMPACT_FLOORS)
    fp = ["Off-topic false-positive rate (share of off-topic queries that still get a product) "
          "at each `min_similarity` (cosine) floor:", "", header, rule]
    rec = ["", f"recall@{LIMIT} at each floor:", "", header, rule]
    for name, cfg in data["configs"].items():
        fp.append(f"| {name} | " + " | ".join(_fmt(r["off_topic_fp_rate"]) for r in cfg["floors"]) + " |")
        rec.append(f"| {name} | " + " | ".join(_fmt(r[f"recall@{LIMIT}"]) for r in cfg["floors"]) + " |")
    return fp + rec


def _separation_table(data):
    lines = [
        "| config | best off-topic cosine | weakest first correct cosine | best off-topic score | "
        "weakest first correct score |",
        "|---|---|---|---|---|",
    ]
    for name, cfg in data["configs"].items():
        c, s = cfg["separation_cosine"], cfg["separation_score"]
        lines.append(
            f"| {name} | {_fmt(c['off_topic_best_similarity_max'])} | "
            f"{_fmt(c['on_topic_first_relevant_similarity_min'])} | "
            f"{_fmt(s['off_topic_best_similarity_max'])} | {_fmt(s['on_topic_first_relevant_similarity_min'])} |"
        )
    return lines


def _by_type_table(data):
    shown = [BEFORE, "dense+excl", "bm25+excl", "hybrid+excl"]
    lines = [
        "recall@3 / MRR per query type, before and with exclusions in each mode:",
        "",
        "| type | n | " + " | ".join(shown) + " |",
        "|---|---|" + "---|" * len(shown),
    ]
    for query_type, row in data["configs"][BEFORE]["by_type"].items():
        cells = []
        for name in shown:
            t = data["configs"][name]["by_type"][query_type]
            cells.append(f"{_fmt(t['recall@3'])} / {_fmt(t['mrr'])}")
        lines.append(f"| {query_type} | {row['n']} | " + " | ".join(cells) + " |")
    return lines


def _parser_lines(data):
    p = data["configs"][AFTER]["parser"]
    lines = [
        f"Exclusion parser ({AFTER}): ruled-out products fully excluded in "
        f"**{p['negations_fully_excluded']} of {p['n_negation_with_excluded']}** negation queries; "
        f"a product labelled relevant was excluded in **{p['relevant_wrongly_excluded']}** queries; "
        f"hedge queries with any exclusion: **{p['hedges_with_exclusion']} of {p['n_hedge']}**.",
        "",
        f"| id | type | query | ruled out (label) | parser excluded | text searched | top {LIMIT} {BEFORE} | "
        f"top {LIMIT} {AFTER} |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in data["queries"]:
        if r["type"] in ("negation", "hedge"):
            lines.append(
                f"| {r['id']} | {r['type']} | {r['query']} | {', '.join(r['excluded']) or '—'} | "
                f"{', '.join(r['parser_excluded']) or '—'} | {r['query_text']} | "
                f"{', '.join(r['top3_by_config'][BEFORE])} | {', '.join(r['top3_by_config'][AFTER]) or '(nothing)'} |"
            )
    return lines


def _misses(data):
    misses = [
        r for r in data["queries"]
        if r["relevant"] and (r["first_relevant_rank"] is None or r["first_relevant_rank"] > LIMIT)
    ]
    if not misses:
        return ["None."]
    lines = [f"| id | type | query | relevant | first relevant rank | top {LIMIT} returned (cosine / score) |",
             "|---|---|---|---|---|---|"]
    for r in misses:
        lines.append(
            f"| {r['id']} | {r['type']} | {r['query']} | {', '.join(r['relevant'])} | "
            f"{_fmt(r['first_relevant_rank'])} | {_top(r['ranking'], score=True)} |"
        )
    return lines


def render_set(data, cfg):
    lines = [
        f"## {data['label']}",
        "",
        f"`{data['file']}`: " + ", ".join(f"{t} {n}" for t, n in data["queries_by_type"].items())
        + f". {data['note']}",
        "",
        "### Every configuration (no floor, top 3)",
        "",
        *_config_table(data),
        "",
        "### Floors",
        "",
        *_floor_tables(data),
        "",
        "### Separation (no floor)",
        "",
        "A floor on a score separates off-topic from on-topic queries cleanly only if the best "
        "off-topic top result scores below the weakest first correct hit.",
        "",
        *_separation_table(data),
        "",
        "### By query type",
        "",
        *_by_type_table(data),
        "",
        "### Negation and hedge queries",
        "",
        *_parser_lines(data),
        "",
        f"### Misses ({AFTER}: on-topic queries with no relevant product in the top {LIMIT})",
        "",
        *_misses(data),
        "",
        f"### Off-topic queries: top result ({AFTER}, cosine / score)",
        "",
        "| id | query | top product |",
        "|---|---|---|",
    ]
    for r in data["queries"]:
        if r["type"] == "off_topic":
            lines.append(f"| {r['id']} | {r['query']} | {_top(r['ranking'], n=1, score=True)} |")
    if "ablation_raw_text" in data:
        lines += [
            "",
            f"### `min_similarity` sweep ({AFTER})",
            "",
            f"| min_similarity | recall@{LIMIT} | MRR | on-topic queries returning nothing | off-topic FP rate |",
            "|---|---|---|---|---|",
        ]
        for row in data["after_sweep"]:
            lines.append(
                f"| {_fmt(row['min_similarity'])} | {_fmt(row[f'recall@{LIMIT}'])} | {_fmt(row['mrr'])} | "
                f"{_fmt(row['on_topic_empty_rate'])} | {_fmt(row['off_topic_fp_rate'])} |"
            )
        lines += [
            "",
            f"Floor the pre-set rule would pick ({cfg['floor_rule']}): "
            f"**{_fmt(data['floor_picked_by_rule'])}**. In-sample: chosen on the queries it is scored on.",
            "",
            "### Ablation: no text normalisation",
            "",
            data["ablation_raw_text"]["note"],
            "",
            "| pipeline | " + " | ".join(f"recall@{k}" for k in cfg["k_values"]) + " | MRR |",
            "|---|" + "---|" * len(cfg["k_values"]) + "---|",
        ]
        for label, row in (("preprocess_text", data["configs"][BEFORE]["no_floor"]),
                           ("raw text", data["ablation_raw_text"]["no_floor"])):
            lines.append(
                f"| {label} | " + " | ".join(_fmt(row[f"recall@{k}"]) for k in cfg["k_values"])
                + f" | {_fmt(row['mrr'])} |"
            )
    return lines


def render_markdown(results):
    cfg = results["config"]
    lines = [
        "# Retrieval eval results",
        "",
        "_Generated by `python -m evals.run_eval` from [`queries.json`](queries.json) and "
        "[`queries-heldout.json`](queries-heldout.json); numbers in [`results.json`](results.json). "
        "Do not edit by hand — CI runs `python -m evals.run_eval --check`._",
        "",
        f"- Catalogue: {cfg['n_products']} products · dense: `{cfg['embedding_model']}`, {cfg['distance']} "
        f"· keyword: {cfg['keyword']} · hybrid: {cfg['fusion']}.",
        f"- Configurations: `dense`, `bm25`, `hybrid`; `+excl` = exclusion parsing on. "
        f"**Before** = `{cfg['before']}` (the API until this change); **after** = `{cfg['after']}` "
        f"(the API default now).",
        f"- Default mode rule ({cfg['default_mode_rule']}) → **{cfg.get('default_mode_by_rule')}**.",
        "- Every query goes through `retrieval.search_detailed`, the function `/similar_products` "
        "calls. Metrics are over the top 3 (the API default `limit`) unless stated. Queries and "
        "judgments are author-written and small: these numbers describe this catalogue, not general "
        "retrieval quality.",
        "- Negation accuracy = share of negation queries with no ruled-out product in the top 3. "
        "Off-topic FP = share of off-topic queries that get at least one product back. In `bm25` "
        "mode only products sharing a word with the query are returned.",
        "",
        "## Summary: before → after",
        "",
        "| set | config | recall@1 | recall@3 | MRR | negation accuracy | off-topic FP (no floor) |",
        "|---|---|---|---|---|---|---|",
    ]
    for set_name in SETS:
        data = results[set_name]
        for tag, name in (("before", cfg["before"]), ("after", cfg["after"])):
            s = data["configs"][name]["no_floor"]
            lines.append(
                f"| {set_name} | {tag}: {name} | {_fmt(s['recall@1'])} | {_fmt(s['recall@3'])} | "
                f"{_fmt(s['mrr'])} | {_fmt(s['negation_accuracy'])} | {_fmt(s['off_topic_fp_rate'])} |"
            )
    lines.append("")
    for set_name in ("heldout", "dev"):
        lines += render_set(results[set_name], cfg) + [""]
    lines += ["## Environment of the committed run", ""]
    for pkg, ver in results.get("environment", {}).items():
        lines.append(f"- {pkg} {ver}")
    return "\n".join(lines) + "\n"


def _compare(new, old, path="", tol=1e-9, errors=None):
    errors = [] if errors is None else errors
    if ".queries" in path:
        tol = SIM_TOLERANCE
    if isinstance(new, dict) and isinstance(old, dict):
        for key in sorted(set(new) | set(old)):
            if key not in new or key not in old:
                errors.append(f"{path}.{key}: present in only one of new/committed")
            else:
                _compare(new[key], old[key], f"{path}.{key}" if path else key, tol, errors)
    elif isinstance(new, list) and isinstance(old, list):
        if len(new) != len(old):
            errors.append(f"{path}: length {len(new)} != committed {len(old)}")
        for i, (a, b) in enumerate(zip(new, old)):
            _compare(a, b, f"{path}[{i}]", tol, errors)
    elif isinstance(new, float) and isinstance(old, (int, float)) and not isinstance(old, bool):
        if abs(new - old) > tol:
            errors.append(f"{path}: {new} != committed {old}")
    elif new != old:
        errors.append(f"{path}: {new!r} != committed {old!r}")
    return errors


def _print_summary(results):
    for set_name in SETS:
        if set_name not in results:
            continue
        for name, cfg in results[set_name]["configs"].items():
            s = cfg["no_floor"]
            print(
                f"{set_name:8} {name:12} recall@1 {_fmt(s['recall@1'])} recall@3 {_fmt(s['recall@3'])} "
                f"MRR {_fmt(s['mrr'])} negation-acc {_fmt(s['negation_accuracy'])} "
                f"off-topic FP {_fmt(s['off_topic_fp_rate'])}"
            )
    if "default_mode_by_rule" in results["config"]:
        print(f"default mode by rule: {results['config']['default_mode_by_rule']} "
              f"(retrieval.DEFAULT_MODE = {retrieval.DEFAULT_MODE})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare against the committed results; exit 1 on drift")
    ap.add_argument("--only", choices=sorted(SETS), help="evaluate one set and print it; writes nothing")
    args = ap.parse_args()

    results = evaluate(only=args.only)
    _print_summary(results)
    if args.only:
        return

    if results["config"]["default_mode_by_rule"] != retrieval.DEFAULT_MODE:
        print(f"retrieval.DEFAULT_MODE ({retrieval.DEFAULT_MODE}) differs from what the pre-set rule "
              f"picks ({results['config']['default_mode_by_rule']})", file=sys.stderr)
        sys.exit(1)

    if args.check:
        committed = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        committed_env = committed.pop("environment", None)
        errors = _compare(results, committed)
        committed["environment"] = committed_env
        if RESULTS_MD.read_text(encoding="utf-8") != render_markdown(committed):
            errors.append("RESULTS.md does not match what results.json renders to (edited by hand?)")
        if errors:
            print(f"DRIFT: {len(errors)} difference(s) from the committed results", file=sys.stderr)
            for e in errors[:50]:
                print("  " + e, file=sys.stderr)
            sys.exit(1)
        print("check OK: results match the committed results.json and RESULTS.md")
        return

    results["environment"] = _environment()
    RESULTS_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    RESULTS_MD.write_text(render_markdown(results), encoding="utf-8")
    print(f"wrote {RESULTS_JSON.relative_to(ROOT)} and {RESULTS_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
