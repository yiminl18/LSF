"""Driver: proxy-only Pareto rule selection for every question.

Mirrors `run_select_all_pareto.py` but uses the proxy-only variant — zero LLM
calls in the selection loop. Writes to `selected_rules_pareto_proxy/`.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

from rule_refinement.cost_profile               import load_or_compute_cost_profile
from rule_refinement.select_rules_pareto_proxy  import run_selection_pareto_proxy

QUERIES_FILE        = "data/financebench/sample_queries.txt"
LABELS_FILE         = "data/financebench/sample/single_cluster/random/sample_doc_labels.json"
PROCESSING_DIR      = "data/financebench/processing"
RULES_BASE_DIR      = "rules/financebench_single_cluster/llm/gpt54/one_shot"
EVAL_MERGE_DIR      = "results/financebench_single_cluster/llm/gpt54/one_shot/eval_merge"
EVAL_INDIVIDUAL_DIR = "results/financebench_single_cluster/llm/gpt54/one_shot/eval_individual"
COST_PROFILE_DIR    = "results/financebench_single_cluster/llm/gpt54/one_shot/cost_profile"
SELECTOR_RUN_DIR    = "results/financebench_single_cluster/llm/gpt54/one_shot/selector_run_pareto_proxy"
OUTPUT_DIR          = "results/financebench_single_cluster/llm/gpt54/one_shot/selected_rules_pareto_proxy"
TAU_SAFETY_FLOOR    = 0.0
KNEE_LAMBDA         = 10.0


def make_slug(q: str) -> str:
    s = q.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60]


def _summarise(result: dict) -> dict:
    return {
        "question":                    result["question"],
        "question_slug":               result["question_slug"],
        "selected_rules":              result["selected_rules"],
        "selector_accuracy":           result["selector_accuracy"],
        "baseline_accuracy":           result["baseline_accuracy"],
        "selected_avg_cost_ratio_sum": result["selected_avg_cost_ratio_sum"],
        "frontier_points":             len(result.get("frontier", [])),
        "knee_point":                  result.get("knee_point"),
        "query_table":                 result.get("query_table"),
    }


questions = [l.strip() for l in open(QUERIES_FILE) if l.strip()]
labels: dict = json.loads(Path(LABELS_FILE).read_text(encoding="utf-8"))
doc_names = [k.replace(".pdf", "") for k in labels.keys()]

doc_map: dict[str, dict] = {}
for doc_name in doc_names:
    path = Path(PROCESSING_DIR) / f"{doc_name}_reconstructed.json"
    if path.exists():
        doc_map[doc_name] = json.loads(path.read_text(encoding="utf-8"))
    else:
        print(f"WARNING: missing {path}")

print(f"Questions: {len(questions)}  |  Docs loaded: {len(doc_map)}")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(SELECTOR_RUN_DIR).mkdir(parents=True, exist_ok=True)
Path(COST_PROFILE_DIR).mkdir(parents=True, exist_ok=True)

summary: list[dict] = []

for question in questions:
    slug        = make_slug(question)
    rule_slug   = f"{slug}_10_llm"
    output_slug = f"{slug}_10"

    out_file = Path(OUTPUT_DIR) / f"{rule_slug}.json"
    if out_file.exists():
        print(f"\nSKIP (exists): {out_file.name}")
        existing = json.loads(out_file.read_text(encoding="utf-8"))
        summary.append(_summarise(existing))
        continue

    rule_folder = Path(RULES_BASE_DIR) / rule_slug
    if not rule_folder.is_dir():
        print(f"\nSKIP (no rule folder): {rule_folder}")
        continue

    eval_merge_path = Path(EVAL_MERGE_DIR) / f"{output_slug}_sampled.json"
    if not eval_merge_path.exists():
        print(f"\nSKIP (no eval_merge): {eval_merge_path}")
        continue

    print(f"\n{'=' * 70}")
    print(f"Question : {question}")
    print(f"Rule slug: {rule_slug}")

    doc_paths = [Path(PROCESSING_DIR) / f"{d}_reconstructed.json" for d in doc_map]
    cache_path = Path(COST_PROFILE_DIR) / f"{rule_slug}.json"
    cost_profile = load_or_compute_cost_profile(
        rules_dir=str(rule_folder),
        doc_paths=doc_paths,
        cache_path=cache_path,
    )
    print(f"  Cost profile: {len(cost_profile)} rules")

    eval_individual_dir = Path(EVAL_INDIVIDUAL_DIR) / rule_slug

    try:
        result = run_selection_pareto_proxy(
            rules_dir=RULES_BASE_DIR,
            eval_merge_path=eval_merge_path,
            eval_individual_dir=eval_individual_dir,
            documents=doc_map,
            question_slug=rule_slug,
            question=question,
            labels=labels,
            cost_profile=cost_profile,
            output_dir=SELECTOR_RUN_DIR,
            tau_safety_floor=TAU_SAFETY_FLOOR,
            knee_lambda=KNEE_LAMBDA,
        )
    except Exception as e:
        import traceback
        print(f"  FATAL ERROR: {e}")
        traceback.print_exc()
        continue

    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    n_points = len(result["frontier"])
    knee = result.get("knee_point")
    knee_str = (
        f"knee=(cost={knee['cost']:.5f}, acc={knee['accuracy_match']:.2f}, n={knee['n_rules']})"
        if knee else "knee=None"
    )
    print(
        f"  → {len(result['selected_rules'])} rules selected  "
        f"cost={result['selected_avg_cost_ratio_sum']:.5f}  "
        f"selector_acc={result['selector_accuracy']:.2f}  "
        f"frontier={n_points}pts  {knee_str}"
    )
    print(f"  Saved: {out_file}")
    summary.append(_summarise(result))


print(f"\n{'Question':<53}  {'Rules':>5}  {'BaseAcc':>7}  {'SelAcc':>7}  {'CostSum':>9}  {'FrontPts':>8}")
print("-" * 100)
for s in summary:
    print(
        f"  {s['question'][:51]:<51}  {len(s['selected_rules']):>5}  "
        f"{s['baseline_accuracy']:>7.2f}  {s['selector_accuracy']:>7.2f}  "
        f"{s['selected_avg_cost_ratio_sum']:>9.5f}  {s['frontier_points']:>8d}"
    )

summary_path = Path(OUTPUT_DIR) / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSummary: {summary_path}")
print(f"Results: {OUTPUT_DIR}/")
