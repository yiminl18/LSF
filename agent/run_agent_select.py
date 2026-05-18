"""Driver: spawn one Claude Code session per question to select rules.

Mirrors the pattern in src/rule_gen_agent_claude.py — calls the `claude -p`
CLI with the task prompt from agent/task_prompt.md, captures stdout, parses
the AGENTIC_SELECTION_DONE summary line.

Reads the question list from data/financebench/sample_queries.txt.
Writes per-question selections to results/.../selected_rules_agent/<slug>.json
(the agent itself writes this file; the driver just monitors).

Usage:
    python agent/run_agent_select.py                   # all 10 questions, opus model
    python agent/run_agent_select.py --slug what_is_the_registrants_telephone_number_10_llm  # single Q
    python agent/run_agent_select.py --budget 20       # limit verify_accuracy calls per question
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
sys.path.insert(0, str(_ROOT))

from tools._paths import (  # noqa: E402
    RULES_BASE_DIR, SELECTED_RULES_AGENT_DIR, AGENT_TRACE_DIR,
)

QUERIES_FILE = _ROOT / "data/financebench/sample_queries.txt"
TASK_PROMPT_FILE = _THIS / "task_prompt.md"

# Model aliases — match rule_gen_agent_claude.py
_MODEL_ALIASES = {
    "opus":   "claude-opus-4-5",
    "opus47": "claude-opus-4-7",
    "sonnet": "claude-sonnet-4-5",
    "haiku":  "claude-haiku-4-5-20251001",
}


def make_slug(question: str) -> str:
    s = question.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60] + "_10_llm"   # matches the rule folder naming convention


def build_prompt(question: str, question_slug: str, budget: int, model: str) -> str:
    template = TASK_PROMPT_FILE.read_text(encoding="utf-8")
    output_path = SELECTED_RULES_AGENT_DIR / f"{question_slug}.json"
    trace_path  = AGENT_TRACE_DIR / f"{question_slug}.jsonl"
    return template.format(
        question=question,
        question_slug=question_slug,
        budget=budget,
        model=model,
        output_path=str(output_path.relative_to(_ROOT)),
        trace_path=str(trace_path.relative_to(_ROOT)),
    )


def run_agent_for_question(
    question:     str,
    question_slug: str,
    budget:       int,
    model:        str,
    timeout:      int = 3600,
    dry_run:      bool = False,
) -> dict:
    """Spawn `claude -p <prompt>` and return the agent's outcome dict."""
    prompt = build_prompt(question, question_slug, budget, model)
    resolved_model = _MODEL_ALIASES.get(model, model)

    if dry_run:
        return {
            "status":      "dry_run",
            "question":    question,
            "question_slug": question_slug,
            "prompt_chars": len(prompt),
            "model":       resolved_model,
        }

    SELECTED_RULES_AGENT_DIR.mkdir(parents=True, exist_ok=True)
    AGENT_TRACE_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        res = subprocess.run(
            ["claude", "--model", resolved_model, "-p", prompt],
            capture_output=True, text=True, cwd=str(_ROOT), timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status":    "timeout",
            "question":  question,
            "question_slug": question_slug,
            "elapsed":   round(time.time() - t0, 1),
            "stdout":    (exc.stdout or "")[-2000:],
            "stderr":    (exc.stderr or "")[-2000:],
        }

    elapsed = round(time.time() - t0, 1)
    summary_line = ""
    for line in (res.stdout or "").splitlines():
        if line.startswith("AGENTIC_SELECTION_DONE"):
            summary_line = line.strip()
            break

    output_json = SELECTED_RULES_AGENT_DIR / f"{question_slug}.json"
    output_data = None
    if output_json.exists():
        try:
            output_data = json.loads(output_json.read_text(encoding="utf-8"))
        except Exception:
            output_data = None

    return {
        "status":       "ok" if res.returncode == 0 else f"exit_{res.returncode}",
        "question":     question,
        "question_slug": question_slug,
        "elapsed_seconds": elapsed,
        "model":        resolved_model,
        "summary_line": summary_line,
        "output_json":  str(output_json),
        "output_data":  output_data,
        "stdout_tail":  (res.stdout or "")[-1000:],
        "stderr_tail":  (res.stderr or "")[-1000:],
    }


def main():
    ap = argparse.ArgumentParser(description="Run the agentic rule selector per question.")
    ap.add_argument("--slug", help="run a single question slug instead of all")
    ap.add_argument("--budget", type=int, default=30,
                    help="max verify_accuracy calls per question (default 30)")
    ap.add_argument("--model", default="opus",
                    help=f"model alias (default opus). Options: {', '.join(_MODEL_ALIASES)}")
    ap.add_argument("--dry-run", action="store_true",
                    help="don't actually spawn claude — print the prompt and exit")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per-question subprocess timeout in seconds (default 3600)")
    args = ap.parse_args()

    # Load question list
    questions = [l.strip() for l in QUERIES_FILE.read_text().splitlines() if l.strip()]
    if args.slug:
        # Map slug back to question text — match by slug-of-question
        questions = [q for q in questions if make_slug(q) == args.slug]
        if not questions:
            print(f"ERROR: no question maps to slug {args.slug}", file=sys.stderr)
            sys.exit(2)

    print(f"Questions to process: {len(questions)}  model={args.model}  budget={args.budget}")

    results = []
    for q in questions:
        slug = make_slug(q)
        print(f"\n{'='*72}\nQuestion: {q}\nSlug:     {slug}")
        if args.dry_run:
            out = run_agent_for_question(q, slug, args.budget, args.model, dry_run=True)
            print(f"  [dry-run] prompt length = {out['prompt_chars']} chars")
        else:
            out = run_agent_for_question(q, slug, args.budget, args.model, timeout=args.timeout)
            print(f"  status={out['status']}  elapsed={out.get('elapsed_seconds')}s")
            if out.get("summary_line"):
                print(f"  {out['summary_line']}")
            if out.get("output_data"):
                d = out["output_data"]
                print(f"  rules={len(d.get('selected_rules',[]))}  match_rate={d.get('match_rate_on_sampled')}")
        results.append(out)

    # Driver summary
    print(f"\n{'='*72}\nDriver summary:")
    for r in results:
        d = r.get("output_data") or {}
        print(f"  {r['question_slug']:<55}  status={r['status']:<12}  "
              f"n_rules={len(d.get('selected_rules', []))}  "
              f"match={d.get('match_rate_on_sampled', '?')}")


if __name__ == "__main__":
    main()
