"""Driver: run cost-optimal rule selection for every question in sample_queries.txt.

Prerequisites (run once before this script):
    python test/run_eval_merge_sampled.py   # produces eval_merge/<slug>_sampled.json
    # Optional — if eval_individual is present, Phase 3 tau check reads from disk:
    python test/run_eval_individual.py      # produces eval_individual/<slug>/<rule>_eval.json

New artefacts written by this script:
    results/.../cost_profile/<rule_slug>.json      # cached Phase 0 token costs
    results/.../selector_run/<rule_slug>/...       # merge outputs during selection
    results/.../selected_rules/<rule_slug>.json    # final selection result
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

from rule_refinement.cost_profile import load_or_compute_cost_profile
from rule_refinement.select_rules import run_selection

QUERIES_FILE        = "data/financebench/sample_queries.txt"
LABELS_FILE         = "data/financebench/sample/single_cluster/random/sample_doc_labels.json"
PROCESSING_DIR      = "data/financebench/processing"
RULES_BASE_DIR      = "rules/financebench/lsf/single_cluster/llm/gpt54/one_shot"
EVAL_MERGE_DIR      = "results/financebench/lsf/single_cluster/llm/gpt54/one_shot/eval_merge"
EVAL_INDIVIDUAL_DIR = "results/financebench/lsf/single_cluster/llm/gpt54/one_shot/eval_individual"
COST_PROFILE_DIR    = "results/financebench/lsf/single_cluster/llm/gpt54/one_shot/cost_profile"
SELECTOR_RUN_DIR    = "results/financebench/lsf/single_cluster/llm/gpt54/one_shot/selector_run"
OUTPUT_DIR          = "results/financebench/lsf/single_cluster/llm/gpt54/one_shot/selected_rules"
TAU                 = 0.20
MODEL_NAME          = "gpt54"


def make_slug(q: str) -> str:
    s = q.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60]


# ── Load questions and labels ──────────────────────────────────────────────────
questions = [l.strip() for l in open(QUERIES_FILE) if l.strip()]
labels: dict = json.loads(Path(LABELS_FILE).read_text(encoding="utf-8"))
doc_names = [k.replace(".pdf", "") for k in labels.keys()]

# Load all documents once
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

# ── Per-question selection ─────────────────────────────────────────────────────
for question in questions:
    slug = make_slug(question)
    rule_slug = f"{slug}_10_llm"     # locates rule folder and used as question_slug in rule_apply_merge
    output_slug = f"{slug}_10"       # used in eval_merge filenames

    out_file = Path(OUTPUT_DIR) / f"{rule_slug}.json"
    if out_file.exists():
        print(f"\nSKIP (exists): {out_file.name}")
        existing = json.loads(out_file.read_text(encoding="utf-8"))
        summary.append({
            "question": existing["question"],
            "question_slug": existing["question_slug"],
            "selected_rules": existing["selected_rules"],
            "selector_accuracy": existing["selector_accuracy"],
            "baseline_accuracy": existing["baseline_accuracy"],
            "selected_avg_cost_ratio_sum": existing["selected_avg_cost_ratio_sum"],
        })
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

    # Phase 0 — cost profile (cached)
    doc_paths = [
        Path(PROCESSING_DIR) / f"{d}_reconstructed.json"
        for d in doc_map
    ]
    cache_path = Path(COST_PROFILE_DIR) / f"{rule_slug}.json"
    cost_profile = load_or_compute_cost_profile(
        rules_dir=str(rule_folder),
        doc_paths=doc_paths,
        cache_path=cache_path,
    )
    print(f"  Cost profile: {len(cost_profile)} rules")

    eval_individual_dir = Path(EVAL_INDIVIDUAL_DIR) / rule_slug

    try:
        result = run_selection(
            rules_dir=RULES_BASE_DIR,
            eval_merge_path=eval_merge_path,
            eval_individual_dir=eval_individual_dir,
            documents=doc_map,
            question_slug=rule_slug,
            question=question,
            labels=labels,
            cost_profile=cost_profile,
            tau=TAU,
            model_name=MODEL_NAME,
            output_dir=SELECTOR_RUN_DIR,
        )
    except Exception as e:
        import traceback
        print(f"  FATAL ERROR: {e}")
        traceback.print_exc()
        continue

    out_file.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"  → {len(result['selected_rules'])} rules selected  "
        f"cost={result['selected_avg_cost_ratio_sum']:.5f}  "
        f"selector_acc={result['selector_accuracy']:.2f}  "
        f"(baseline={result['baseline_accuracy']:.2f})"
    )
    print(f"  Saved: {out_file}")

    summary.append({
        "question": result["question"],
        "question_slug": result["question_slug"],
        "selected_rules": result["selected_rules"],
        "selector_accuracy": result["selector_accuracy"],
        "baseline_accuracy": result["baseline_accuracy"],
        "selected_avg_cost_ratio_sum": result["selected_avg_cost_ratio_sum"],
    })

# ── Summary table ──────────────────────────────────────────────────────────────
print(f"\n{'Question':<55}  {'Rules':>5}  {'BaseAcc':>7}  {'SelAcc':>7}  {'CostSum':>9}")
print("-" * 88)
for s in summary:
    print(
        f"  {s['question'][:53]:<53}  {len(s['selected_rules']):>5}  "
        f"{s['baseline_accuracy']:>7.2f}  {s['selector_accuracy']:>7.2f}  "
        f"{s['selected_avg_cost_ratio_sum']:>9.5f}"
    )

summary_path = Path(OUTPUT_DIR) / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSummary: {summary_path}")
print(f"Results: {OUTPUT_DIR}/")
