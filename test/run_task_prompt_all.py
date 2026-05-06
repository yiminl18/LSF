"""Run task_prompt_rule_gen.py for all questions in a queries file, skipping existing."""

from __future__ import annotations

import argparse
import json as _json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "test"))

from task_prompt_rule_gen import run

# ── CLI args ──────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Run claude agent rule gen for all queries.")
parser.add_argument("--labels-file",    default="data/financebench/sample_doc_labels.json")
parser.add_argument("--queries-file",   default=None,
                    help="Path to queries txt. If omitted, questions are derived from labels-file.")
parser.add_argument("--processing-dir", default="data/financebench/processing")
parser.add_argument("--rules-dir",      default="rules/claude_opus/financebench")
parser.add_argument("--model",          default="opus")
parser.add_argument("--log-dir",        default="logs/claude_agent")
args = parser.parse_args()

LABELS_FILE    = args.labels_file
PROCESSING_DIR = args.processing_dir
RULES_DIR      = args.rules_dir
MODEL          = args.model
LOG_DIR        = ROOT / args.log_dir

LOG_DIR.mkdir(parents=True, exist_ok=True)

# Derive doc names from labels file
_labels = _json.loads((ROOT / LABELS_FILE).read_text())
DOCS = [k.replace(".pdf", "").replace(".PDF", "") for k in _labels.keys()]

# Derive questions: from queries file if given, else from labels file
if args.queries_file:
    QUERIES_FILE = ROOT / args.queries_file
    questions = [l.strip() for l in QUERIES_FILE.read_text().splitlines() if l.strip()]
else:
    # Collect all questions that appear in at least one doc in the labels file
    q_set: dict[str, None] = {}
    for doc_qs in _labels.values():
        for q in doc_qs:
            q_set.setdefault(q, None)
    questions = list(q_set.keys())


def slug(q: str) -> str:
    return re.sub(r"[^\w]", "_", q.lower())[:60].rstrip("_")


print(f"Questions: {len(questions)}, Docs: {len(DOCS)}", flush=True)

for i, question in enumerate(questions, 1):
    q_slug = slug(question)
    rules_subdir = ROOT / RULES_DIR / q_slug
    summary_json = ROOT / RULES_DIR / f"{q_slug}_rule_gen.json"

    if rules_subdir.exists() or summary_json.exists():
        print(f"[{i}/{len(questions)}] SKIP {q_slug}", flush=True)
        continue

    print(f"[{i}/{len(questions)}] START {q_slug}", flush=True)
    log_file = LOG_DIR / f"{q_slug}.txt"
    t0 = time.monotonic()
    try:
        output = run(
            question=question,
            docs=DOCS,
            labels_file=LABELS_FILE,
            processing_dir=PROCESSING_DIR,
            rules_dir=RULES_DIR,
            model=MODEL,
            cwd=str(ROOT),
        )
        elapsed = time.monotonic() - t0
        log_file.write_text(output, encoding="utf-8")
        print(f"[{i}/{len(questions)}] DONE  {q_slug}  ({elapsed:.0f}s)", flush=True)
    except Exception as e:
        elapsed = time.monotonic() - t0
        log_file.write_text(str(e), encoding="utf-8")
        print(f"[{i}/{len(questions)}] ERROR {q_slug}  ({elapsed:.0f}s): {e}", flush=True)

print("All done.", flush=True)
