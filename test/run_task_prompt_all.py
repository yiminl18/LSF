"""Run task_prompt_rule_gen.py for all questions in sample_queries.txt, skipping existing."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "test"))

from task_prompt_rule_gen import run

QUERIES_FILE   = ROOT / "data/financebench/sample_queries.txt"
LABELS_FILE    = "data/financebench/sample_doc_labels.json"
PROCESSING_DIR = "data/financebench/processing"
RULES_DIR      = "rules/agent/financebench_agent"
MODEL          = "opus"
LOG_DIR        = ROOT / "logs" / "claude_agent"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# Derive doc names from labels file
import json as _json
_labels = _json.loads((ROOT / LABELS_FILE).read_text())
DOCS = [k.replace(".pdf", "").replace(".PDF", "") for k in _labels.keys()]

def slug(q: str) -> str:
    return re.sub(r"[^\w]", "_", q.lower())[:60].rstrip("_")

questions = [l.strip() for l in QUERIES_FILE.read_text().splitlines() if l.strip()]
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
