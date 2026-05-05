"""Apply rule_apply_merge for every question on both sampled and unsampled docs."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

from rule_apply_merge import rule_apply_merge

QUERIES_FILE     = "data/financebench/sample_queries.txt"
SAMPLE_LABELS    = "data/financebench/sample_doc_labels.json"
UNSAMPLED_LABELS = "data/financebench/unsampled_doc_labels.json"
PROCESSING_DIR   = "data/financebench/processing"
RULES_DIR        = "rules/financebench"
SAMPLED_OUT_DIR  = "results/financebench/rule_run/merge"
UNSAMPLED_OUT_DIR= "results/financebench/rule_run/merge_unsampled"


def make_slug(question: str) -> str:
    slug = question.lower()
    slug = re.sub(r"[^\w\s]", "", slug)
    slug = re.sub(r"\s+", "_", slug)
    return slug[:60]


def _already_processed(output_dir: str, question_slug: str, rule_set_slug: str) -> set[str]:
    out_path = Path(output_dir) / question_slug / f"{rule_set_slug}_merge.json"
    if not out_path.exists():
        return set()
    try:
        records = json.loads(out_path.read_text(encoding="utf-8"))
        return {r["doc_name"] for r in records if "doc_name" in r}
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# STEP 2 — Load questions
# ---------------------------------------------------------------------------
questions = [line.strip() for line in open(QUERIES_FILE) if line.strip()]
print(f"Questions loaded: {len(questions)}")

# ---------------------------------------------------------------------------
# STEP 4 — Load label files
# ---------------------------------------------------------------------------
sample_labels   = json.loads(Path(SAMPLE_LABELS).read_text(encoding="utf-8"))
unsampled_labels= json.loads(Path(UNSAMPLED_LABELS).read_text(encoding="utf-8"))
print(f"Sampled docs:   {len(sample_labels)}")
print(f"Unsampled docs: {len(unsampled_labels)}")

n_questions_processed = 0
n_questions_skipped   = 0

# ---------------------------------------------------------------------------
# STEP 3 & 5 & 6 — Per question
# ---------------------------------------------------------------------------
for question in questions:
    slug          = make_slug(question)
    question_slug = f"{slug}_10"
    rule_folder   = f"{RULES_DIR}/{question_slug}"

    if not os.path.isdir(rule_folder):
        print(f"\nWARNING: no rule folder for '{question}' at {rule_folder}, skipping")
        n_questions_skipped += 1
        continue

    rule_names = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(rule_folder)
        if f.startswith("rule_") and f.endswith(".py")
    )
    if not rule_names:
        print(f"\nWARNING: rule folder empty: {rule_folder}, skipping")
        n_questions_skipped += 1
        continue

    print(f"\nQuestion: {question}")
    print(f"  slug: {question_slug}   rules: {len(rule_names)}")

    rule_set_slug = "__".join(sorted(rule_names))[:120]

    def run_on_label_set(labels_dict: dict, output_dir: str, split_tag: str):
        already_done = _already_processed(output_dir, question_slug, rule_set_slug)
        doc_names = [k.replace(".pdf", "") for k in labels_dict.keys()]
        completed = skipped = 0
        for doc_name in doc_names:
            if doc_name in already_done:
                skipped += 1
                continue
            doc_path = f"{PROCESSING_DIR}/{doc_name}_reconstructed.json"
            if not os.path.exists(doc_path):
                print(f"    SKIP (missing): {doc_path}")
                skipped += 1
                continue
            try:
                document = json.loads(Path(doc_path).read_text(encoding="utf-8"))
                result = rule_apply_merge(
                    document=document,
                    rule_names=rule_names,
                    question_slug=question_slug,
                    question=question,
                    output_dir=output_dir,
                )
                print(f"    [{split_tag}] {doc_name}: {str(result['predicted_answer'])[:50]}")
                completed += 1
            except Exception as e:
                print(f"    ERROR {doc_name}: {e}")
                skipped += 1
        return completed, skipped

    try:
        c, s = run_on_label_set(sample_labels, SAMPLED_OUT_DIR, "sampled")
        print(f"  sampled:   {c} completed, {s} skipped")

        c, s = run_on_label_set(unsampled_labels, UNSAMPLED_OUT_DIR, "unsampled")
        print(f"  unsampled: {c} completed, {s} skipped")

        n_questions_processed += 1
    except Exception as e:
        print(f"  ERROR processing question '{question}': {e}")
        n_questions_skipped += 1

# ---------------------------------------------------------------------------
# STEP 7 — Summary
# ---------------------------------------------------------------------------
print()
print("=== Run Complete ===")
print(f"Questions processed:               {n_questions_processed}")
print(f"Questions skipped (no rule folder): {n_questions_skipped}")
print(f"Output dirs:")
print(f"  sampled   → {SAMPLED_OUT_DIR}/")
print(f"  unsampled → {UNSAMPLED_OUT_DIR}/")
