"""Evaluate all rules under what_is_the_registrants_exact_name_10 and print a summary."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

from eval_rule import eval_rule
from rule_apply_individual import rule_apply_individual

QUESTION = "What is the registrant's exact name?"
QUESTION_SLUG = "what_is_the_registrants_exact_name_10"
RULES_DIR = _ROOT / "rules/financebench_single_cluster/llm/gpt54/one_shot/what_is_the_registrants_exact_name_10_llm"
LABELS_FILE = "data/financebench/correct_labels.json"

# ---------------------------------------------------------------------------
# Load doc_names from most recent rule_gen result matching the question slug
# ---------------------------------------------------------------------------
import glob as _glob

gen_files = sorted(_glob.glob("results/financebench_single_cluster/llm/gpt54/one_shot/rule_gen/*what_is_the_registrants_exact_name*.json"))
if not gen_files:
    print("ERROR: no rule_gen result files found")
    sys.exit(1)
latest = json.loads(Path(gen_files[-1]).read_text(encoding="utf-8"))
doc_names: list[str] = latest["doc_names"]
print(f"doc_names ({len(doc_names)}): {doc_names}")
print()

# ---------------------------------------------------------------------------
# Pre-run: call rule_apply_individual for any (rule, doc) pairs missing predictions
# ---------------------------------------------------------------------------
rule_files = sorted(_glob.glob(str(RULES_DIR / "rule_*.py")))
print(f"Rules to evaluate: {len(rule_files)}")
print("Pre-running rule_apply_individual for missing predictions...")
print()

for rule_file in rule_files:
    rule_name = Path(rule_file).stem
    pred_file = _ROOT / f"results/financebench_single_cluster/llm/gpt54/one_shot/rule_run_individual/{QUESTION_SLUG}/{rule_name}_individual.json"

    existing_docs: set[str] = set()
    if pred_file.exists():
        records = json.loads(pred_file.read_text(encoding="utf-8"))
        existing_docs = {r["doc_name"] for r in records}

    missing = [d for d in doc_names if d not in existing_docs]
    if not missing:
        continue

    print(f"  {rule_name}: running on {len(missing)} docs")
    for doc_name in missing:
        doc_path = _ROOT / "data/financebench/processing" / f"{doc_name}_reconstructed.json"
        if not doc_path.exists():
            print(f"    skip {doc_name}: no JSON")
            continue
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        try:
            rule_apply_individual(
                document=doc,
                rule_name=rule_name,
                question_slug=QUESTION_SLUG,
                question=QUESTION,
                rules_dir="rules/financebench_single_cluster/llm/gpt54/one_shot",
                output_dir="results/financebench_single_cluster/llm/gpt54/one_shot/rule_run_individual",
            )
        except Exception as exc:
            print(f"    error on {doc_name}: {exc}")

print()

# ---------------------------------------------------------------------------
# Evaluate all rules
# ---------------------------------------------------------------------------
results: list[dict] = []

for rule_file in rule_files:
    rule_name = Path(rule_file).stem
    print(f"Evaluating {rule_name} ...")
    try:
        result = eval_rule(
            rule_name=rule_name,
            doc_names=doc_names,
            question=QUESTION,
            question_slug=QUESTION_SLUG,
            labels_file=LABELS_FILE,
        )
        row = {
            "rule_name": rule_name,
            "accuracy": result["accuracy"],
            "avg_cost_ratio": result["avg_cost_ratio"],
            "avg_rule_apply_latency_seconds": result["avg_rule_apply_latency_seconds"],
            "avg_judge_latency_seconds": result["avg_judge_latency_seconds"],
            "total_eval_latency_seconds": result["total_eval_latency_seconds"],
        }
        results.append(row)
        print(
            f"  accuracy={result['accuracy']:.2f}  "
            f"cost_ratio={result['avg_cost_ratio']:.4f}  "
            f"rule_apply={result['avg_rule_apply_latency_seconds']:.1f}s  "
            f"judge={result['avg_judge_latency_seconds']:.1f}s  "
            f"total={result['total_eval_latency_seconds']:.1f}s"
        )
    except Exception as exc:
        print(f"  ERROR: {exc}")

# ---------------------------------------------------------------------------
# Summary table sorted by accuracy descending
# ---------------------------------------------------------------------------
results.sort(key=lambda r: (-r["accuracy"], r["avg_cost_ratio"]))

print()
print("=== Summary ===")
header = f"{'Rule':<62}  {'acc':>4}  {'cost':>7}  {'rule_apply':>10}  {'judge':>7}  {'total':>7}"
print(header)
print("-" * len(header))
for r in results:
    print(
        f"{r['rule_name']:<62}  "
        f"{r['accuracy']:>4.2f}  "
        f"{r['avg_cost_ratio']:>7.4f}  "
        f"{r['avg_rule_apply_latency_seconds']:>9.1f}s  "
        f"{r['avg_judge_latency_seconds']:>6.1f}s  "
        f"{r['total_eval_latency_seconds']:>6.1f}s"
    )
