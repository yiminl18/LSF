"""Generate rules for all 30 questions, skipping any that already have a rule folder."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

from rule_gen_llm_coarse import rule_gen_llm_coarse

QUERIES_FILE   = "data/financebench/sample_queries.txt"
LABELS_FILE    = "data/financebench/sample_doc_labels.json"
PROCESSING_DIR = "data/financebench/processing"
RULES_DIR      = "rules/financebench_single_cluster/llm/gpt54/one_shot"
OUTPUT_DIR     = "results/financebench_single_cluster/llm/gpt54/one_shot/rule_gen"

# ---------------------------------------------------------------------------
# STEP 1 — Load questions
# ---------------------------------------------------------------------------
questions = [line.strip() for line in open(QUERIES_FILE) if line.strip()]
print(f"Questions loaded: {len(questions)}")

# ---------------------------------------------------------------------------
# STEP 2 — Slug function
# ---------------------------------------------------------------------------
def make_slug(question: str) -> str:
    slug = question.lower()
    slug = re.sub(r"[^\w\s]", "", slug)
    slug = re.sub(r"\s+", "_", slug)
    return slug[:60]

# ---------------------------------------------------------------------------
# STEP 3 — Load labels
# ---------------------------------------------------------------------------
labels: dict = json.loads(Path(LABELS_FILE).read_text(encoding="utf-8"))
doc_names_pdf = list(labels.keys())
print(f"Training docs: {len(doc_names_pdf)}")

# ---------------------------------------------------------------------------
# STEP 4 — Load document JSONs
# ---------------------------------------------------------------------------
documents = []
for pdf_key in doc_names_pdf:
    doc_name = pdf_key.replace(".pdf", "")
    doc_path = f"{PROCESSING_DIR}/{doc_name}_reconstructed.json"
    if os.path.exists(doc_path):
        documents.append(json.loads(Path(doc_path).read_text(encoding="utf-8")))
    else:
        print(f"WARNING: missing {doc_path}, skipping")
print(f"Documents loaded: {len(documents)}")
print()

# ---------------------------------------------------------------------------
# STEP 5 — Generate rules per question
# ---------------------------------------------------------------------------
n_skipped = 0
n_processed = 0
total_rules = 0

for i, question in enumerate(questions, 1):
    slug = make_slug(question)
    rule_folder = f"{RULES_DIR}/{slug}_10"

    if os.path.isdir(rule_folder):
        print(f"[{i:02d}/30] SKIP (exists): {slug}_10")
        n_skipped += 1
        continue

    print(f"\n[{i:02d}/30] {question}")
    print(f"  slug: {slug}")

    ground_truth = {
        pdf_key: qa.get(question)
        for pdf_key, qa in labels.items()
        if question in qa
    }

    try:
        result = rule_gen_llm_coarse(
            documents=documents,
            question=question,
            ground_truth=ground_truth,
            output_dir=OUTPUT_DIR,
            rules_dir=RULES_DIR,
        )
        n_rules = len(result["rules"])
        total_rules += n_rules
        n_processed += 1
        print(f"  generated {n_rules} rules → {rule_folder}")
        print(f"  latency: {result['latency_seconds']:.1f}s  "
              f"tokens in/out: {result['input_tokens']}/{result['output_tokens']}")
    except Exception as exc:
        print(f"  ERROR: {exc}")

# ---------------------------------------------------------------------------
# STEP 6 — Summary
# ---------------------------------------------------------------------------
print()
print("=== Rule Generation Complete ===")
print(f"Questions processed:               {n_processed} / {len(questions)}")
print(f"Questions skipped (already exist): {n_skipped}")
print(f"Total rules generated:             {total_rules}")
