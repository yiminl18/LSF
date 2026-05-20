"""Evaluate every rule individually for every mix query on the 18 sampled multi-cluster docs."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

from rule_apply_individual import rule_apply_individual
from eval_rule import eval_rule

QUERIES_FILE   = "data/financebench/mix_doc_queries.txt"
LABELS_FILE    = "data/financebench/sample/multi_cluster/random/sample_doc_labels.json"
PROCESSING_DIR = "data/financebench/processing"
RULES_BASE_DIR = "rules/financebench_multi_clusters/llm/gpt54/one_shot"
RULE_RUN_DIR   = "results/financebench_multi_clusters/llm/gpt54/one_shot/rule_run_individual"
OUTPUT_DIR     = "results/financebench_multi_clusters/llm/gpt54/one_shot/eval_individual"


def make_slug(q: str) -> str:
    s = q.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60]


# ── Load questions and labels ──────────────────────────────────────────────────
questions = [l.strip() for l in open(QUERIES_FILE) if l.strip()]
labels: dict[str, dict] = json.loads(Path(LABELS_FILE).read_text(encoding="utf-8"))
# Only use docs that have reconstructed JSONs
doc_names = [
    k.replace(".pdf", "") for k in labels
    if (Path(PROCESSING_DIR) / f"{k.replace('.pdf', '')}_reconstructed.json").exists()
]
print(f"Questions: {len(questions)}  |  Docs: {len(doc_names)}\n")

Path(RULE_RUN_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ── Main loop ──────────────────────────────────────────────────────────────────
for question in questions:
    slug        = make_slug(question)
    rule_slug   = f"{slug}_18_llm"
    rule_folder = Path(RULES_BASE_DIR) / rule_slug

    if not rule_folder.is_dir():
        print(f"SKIP (no rule folder): {rule_folder}")
        continue

    rule_names = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(rule_folder)
        if f.startswith("rule_") and f.endswith(".py")
    )
    if not rule_names:
        print(f"SKIP (empty): {rule_folder}")
        continue

    print(f"\nQuestion: {question}  ({len(rule_names)} rules)")

    # ── Apply each rule to each doc ────────────────────────────────────────────
    for rule_name in rule_names:
        run_file = Path(RULE_RUN_DIR) / rule_slug / f"{rule_name}_individual.json"
        already_done: set[str] = set()
        if run_file.exists():
            try:
                existing = json.loads(run_file.read_text(encoding="utf-8"))
                already_done = {r.get("doc_name", "") for r in existing}
            except Exception:
                pass

        for doc_name in doc_names:
            if doc_name in already_done:
                continue
            doc_path = Path(PROCESSING_DIR) / f"{doc_name}_reconstructed.json"
            if not doc_path.exists():
                print(f"  WARNING: missing {doc_path}")
                continue
            try:
                document = json.loads(doc_path.read_text(encoding="utf-8"))
                rule_apply_individual(
                    document=document,
                    rule_name=rule_name,
                    question_slug=rule_slug,
                    question=question,
                    rules_dir=RULES_BASE_DIR,
                    output_dir=RULE_RUN_DIR,
                )
            except Exception as e:
                print(f"  ERROR apply {rule_name} / {doc_name}: {e}")

    # ── Judge predictions and write eval files ─────────────────────────────────
    for rule_name in rule_names:
        eval_file = Path(OUTPUT_DIR) / rule_slug / f"{rule_name}_eval.json"
        if eval_file.exists():
            continue
        try:
            eval_rule(
                rule_name=rule_name,
                doc_names=doc_names,
                question=question,
                question_slug=rule_slug,
                rule_run_dir=RULE_RUN_DIR,
                processing_dir=PROCESSING_DIR,
                labels_file=LABELS_FILE,
                output_dir=OUTPUT_DIR,
            )
            print(f"  eval done: {rule_name}")
        except Exception as e:
            print(f"  ERROR eval {rule_name}: {e}")

# ── Summary table ──────────────────────────────────────────────────────────────
print(f"\n{'=== Individual Rule Eval Summary ==='}")
header = f"{'Question':<48}  {'Rules':>5}  {'Best Rule':<40}  {'BestAcc':>7}  {'AvgCost':>9}"
print(header)
print("-" * len(header))

for question in questions:
    slug      = make_slug(question)
    rule_slug = f"{slug}_18_llm"
    eval_dir  = Path(OUTPUT_DIR) / rule_slug

    if not eval_dir.is_dir():
        continue

    eval_files = sorted(eval_dir.glob("*_eval.json"))
    if not eval_files:
        continue

    results = []
    for ef in eval_files:
        try:
            d = json.loads(ef.read_text(encoding="utf-8"))
            results.append(d)
        except Exception:
            pass

    if not results:
        continue

    best     = max(results, key=lambda r: r.get("accuracy", 0))
    avg_cost = sum(r.get("avg_cost_ratio", 0) for r in results) / len(results)

    print(
        f"  {question[:46]:<46}  {len(results):>5}  "
        f"{best['rule_name'][:38]:<38}  {best.get('accuracy', 0):>7.2f}  {avg_cost:>9.5f}"
    )

print(f"\nRule runs:  {RULE_RUN_DIR}/")
print(f"Eval files: {OUTPUT_DIR}/")
