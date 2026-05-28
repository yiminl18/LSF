"""Driver: spawn one Codex session per question to select rules.

Codex equivalent of `src/rule_refine/agentic.py`. Uses the same task prompt
(`src/rule_refine/agentic_task_prompt.md`), the same per-question budget, the same output
contract (`AGENTIC_SELECTION_DONE …` summary line + `selected_rules_agent/<slug>.json`).
The only difference is the agent backbone: `codex exec` with gpt-5.4 instead of
`claude -p` with Claude Opus.

There is no Claude wrapper layer; this driver spawns one Codex agent session
per question and that session does all rule-selection work end to end.

Usage:
    python src/rule_refine/agentic_codex.py                                  # all questions, gpt54
    python src/rule_refine/agentic_codex.py --slug what_is_the_..._10_llm    # one question
    python src/rule_refine/agentic_codex.py --model gpt54mini                # cheaper inner agent
    python src/rule_refine/agentic_codex.py --budget 20                      # cap verify_accuracy calls

Requires `AZURE_OPENAI_API_KEY` in the environment (extract from the YAML key
file — see docs/codex_setup.md).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))

from tools._paths import (  # noqa: E402
    RULES_BASE_DIR,
    SELECTED_RULES_AGENT_DIR as _DEFAULT_OUT_DIR,
    AGENT_TRACE_DIR          as _DEFAULT_TRACE_DIR,
)

QUERIES_FILE     = _ROOT / "data/financebench/sample_queries.txt"
TASK_PROMPT_FILE = _THIS / "agentic_task_prompt.md"

# Output dirs are module-level so build_prompt() can reference them; CLI may override.
SELECTED_RULES_AGENT_DIR = _DEFAULT_OUT_DIR
AGENT_TRACE_DIR          = _DEFAULT_TRACE_DIR

# Codex model aliases
_MODEL_ALIASES = {
    "gpt54":         "gpt-5.4",
    "gpt54mini":     "gpt-5.4-mini",
    "gpt-5.4":       "gpt-5.4",
    "gpt-5.4-mini":  "gpt-5.4-mini",
}


def make_slug(question: str) -> str:
    s = question.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60] + "_10_llm"   # matches the rule folder naming convention


def build_prompt(question: str, question_slug: str, budget: int, model: str) -> str:
    """Build the prompt by formatting src/rule_refine/agentic_task_prompt.md.

    Same template + same placeholders as the Claude driver — the agent reads
    the prompt and acts; the only difference at run-time is the underlying CLI.
    """
    template = TASK_PROMPT_FILE.read_text(encoding="utf-8")
    output_path = SELECTED_RULES_AGENT_DIR / f"{question_slug}.json"
    trace_path  = AGENT_TRACE_DIR / f"{question_slug}.jsonl"
    return template.format(
        question      = question,
        question_slug = question_slug,
        budget        = budget,
        model         = model,
        output_path   = str(output_path.relative_to(_ROOT)),
        trace_path    = str(trace_path.relative_to(_ROOT)),
    )


def _parse_codex_events(jsonl_text: str) -> dict:
    """Parse the codex --json event stream for usage totals + summary line."""
    usage_totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_output_tokens": 0,
    }
    thread_id     = None
    error_message = None
    event_count   = 0
    last_agent_message = ""

    for raw_line in jsonl_text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        event_count += 1
        et = event.get("type")
        if et == "thread.started":
            thread_id = event.get("thread_id")
        elif et == "turn.completed":
            usage = event.get("usage") or {}
            for k in usage_totals:
                usage_totals[k] += int(usage.get(k) or 0)
        elif et == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message":
                last_agent_message = item.get("text", "") or last_agent_message
        elif et in {"error", "turn.failed"}:
            error_message = event.get("message") or str(event.get("error") or "")

    return {
        "usage_totals":       usage_totals,
        "thread_id":          thread_id,
        "error_message":      error_message,
        "event_count":        event_count,
        "last_agent_message": last_agent_message,
    }


def _clean_stderr(stderr: str) -> str:
    lines = [l for l in stderr.splitlines() if l.strip() != "Reading additional input from stdin..."]
    return "\n".join(lines).strip()[:4000]


def run_agent_for_question(
    question:      str,
    question_slug: str,
    budget:        int,
    model:         str,
    timeout:       int = 5400,
    dry_run:       bool = False,
) -> dict:
    """Spawn `codex exec <prompt>` and return the agent's outcome dict."""
    prompt = build_prompt(question, question_slug, budget, model)
    resolved_model = _MODEL_ALIASES.get(model, model)

    if dry_run:
        return {
            "status":        "dry_run",
            "question":      question,
            "question_slug": question_slug,
            "prompt_chars":  len(prompt),
            "model":         resolved_model,
        }

    SELECTED_RULES_AGENT_DIR.mkdir(parents=True, exist_ok=True)
    AGENT_TRACE_DIR.mkdir(parents=True, exist_ok=True)

    codex_bin = shutil.which("codex")
    if not codex_bin:
        return {
            "status":        "error",
            "question":      question,
            "question_slug": question_slug,
            "error_message": "codex CLI not found on PATH",
        }

    # Codex stores its final message here (last assistant turn after all tool calls)
    last_message_path = AGENT_TRACE_DIR / f"{question_slug}.codex.last.txt"

    cmd = [
        codex_bin,
        "--ask-for-approval", "never",
        "exec",
        "--json", "--color", "never",
        "--model", resolved_model,
        "--cd", str(_ROOT),
        "--sandbox", "danger-full-access",
        "--output-last-message", str(last_message_path),
        prompt,
    ]

    t0 = time.time()
    try:
        res = subprocess.run(
            cmd, input="", capture_output=True, text=True,
            cwd=str(_ROOT), timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status":            "timeout",
            "question":          question,
            "question_slug":     question_slug,
            "wallclock_seconds": round(time.time() - t0, 1),
            "stdout":            (exc.stdout or "")[-2000:] if exc.stdout else "",
            "stderr":            (exc.stderr or "")[-2000:] if exc.stderr else "",
        }

    wallclock = round(time.time() - t0, 1)

    # Persist the raw JSONL event log next to the agent_trace JSONL
    jsonl_text = (res.stdout or "").strip()
    codex_log_path = AGENT_TRACE_DIR / f"{question_slug}.codex.jsonl"
    codex_log_path.write_text(jsonl_text + ("\n" if jsonl_text else ""), encoding="utf-8")

    parsed = _parse_codex_events(jsonl_text)
    last_message = parsed["last_agent_message"]
    if not last_message and last_message_path.exists():
        last_message = last_message_path.read_text(encoding="utf-8", errors="ignore")

    summary_line = ""
    for line in (last_message or "").splitlines():
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

    usage = parsed["usage_totals"]
    return {
        "status":              "ok" if res.returncode == 0 else f"exit_{res.returncode}",
        "question":            question,
        "question_slug":       question_slug,
        "wallclock_seconds":   wallclock,
        "model":               resolved_model,
        "summary_line":        summary_line,
        "codex_input_tokens":  usage["input_tokens"],
        "codex_output_tokens": usage["output_tokens"],
        "codex_cached_tokens": usage["cached_input_tokens"],
        "codex_reasoning_tokens": usage["reasoning_output_tokens"],
        "codex_thread_id":     parsed["thread_id"],
        "codex_event_count":   parsed["event_count"],
        "codex_error_message": parsed["error_message"],
        "output_json":         str(output_json),
        "output_data":         output_data,
        "codex_log_path":      str(codex_log_path),
        "codex_last_message_path": str(last_message_path),
        "stdout_tail":         (last_message or "")[-1000:],
        "stderr_tail":         _clean_stderr(res.stderr or "")[-1000:],
    }


def main():
    ap = argparse.ArgumentParser(description="Run the agentic rule selector (Codex backbone) per question.")
    ap.add_argument("--slug", help="run a single question slug instead of all")
    ap.add_argument("--budget", type=int, default=30,
                    help="max verify_accuracy calls per question (default 30)")
    ap.add_argument("--model", default="gpt54",
                    help=f"model alias (default gpt54). Options: {', '.join(_MODEL_ALIASES)}")
    ap.add_argument("--dry-run", action="store_true",
                    help="don't actually spawn codex — print the prompt and exit")
    ap.add_argument("--timeout", type=int, default=5400,
                    help="per-question subprocess timeout in seconds (default 5400 = 90m)")
    ap.add_argument("--out-dir",   default=None,
                    help="override SELECTED_RULES_AGENT_DIR (per-question <slug>.json land here)")
    ap.add_argument("--trace-dir", default=None,
                    help="override AGENT_TRACE_DIR (codex JSONL + last-message logs land here)")
    args = ap.parse_args()

    global SELECTED_RULES_AGENT_DIR, AGENT_TRACE_DIR
    if args.out_dir:
        SELECTED_RULES_AGENT_DIR = Path(args.out_dir).resolve()
    if args.trace_dir:
        AGENT_TRACE_DIR = Path(args.trace_dir).resolve()

    questions = [l.strip() for l in QUERIES_FILE.read_text().splitlines() if l.strip()]
    if args.slug:
        questions = [q for q in questions if make_slug(q) == args.slug]
        if not questions:
            print(f"ERROR: no question maps to slug {args.slug}", file=sys.stderr)
            sys.exit(2)

    if not os.environ.get("AZURE_OPENAI_API_KEY"):
        print(
            "WARNING: AZURE_OPENAI_API_KEY is not set. Codex will 401 against Azure.\n"
            "  Extract the api_key from the YAML key file before invoking — see docs/codex_setup.md.",
            file=sys.stderr,
        )

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
            print(f"  status={out['status']}  wall={out.get('wallclock_seconds')}s")
            if out.get("summary_line"):
                print(f"  {out['summary_line']}")
            if out.get("output_data"):
                d = out["output_data"]
                print(f"  rules={len(d.get('selected_rules', []))}  "
                      f"match_rate={d.get('match_rate_on_sampled')}")
        results.append(out)

    # Driver summary
    print(f"\n{'='*72}\nDriver summary:")
    total_codex_in = total_codex_out = 0
    total_tool_in  = total_tool_out  = total_tool_calls = 0
    total_wallclock = 0.0
    for r in results:
        d = r.get("output_data") or {}
        codex_in  = (r.get("codex_input_tokens", 0) or 0) + (r.get("codex_cached_tokens", 0) or 0)
        codex_out = r.get("codex_output_tokens", 0) or 0
        total_codex_in  += codex_in
        total_codex_out += codex_out
        total_tool_in    += d.get("tool_input_tokens", 0) or 0
        total_tool_out   += d.get("tool_output_tokens", 0) or 0
        total_tool_calls += d.get("tool_llm_calls", 0) or 0
        total_wallclock  += r.get("wallclock_seconds", 0) or 0
        print(f"  {r['question_slug']:<50}  status={r['status']:<10}  "
              f"rules={len(d.get('selected_rules', [])):>2}  "
              f"match={str(d.get('match_rate_on_sampled', '?')):>5}  "
              f"codex_in={codex_in:>7}  codex_out={codex_out:>5}  "
              f"tool_in={d.get('tool_input_tokens', 0):>6}  "
              f"tool_out={d.get('tool_output_tokens', 0):>4}  "
              f"wall={r.get('wallclock_seconds', 0):>5.0f}s")

    print(f"\n{'='*72}\nTotals across {len(results)} question(s):")
    print(f"  Codex tokens:      in={total_codex_in:,}  out={total_codex_out:,}")
    print(f"  Tool gpt54 tokens: in={total_tool_in:,}  out={total_tool_out:,}  (calls={total_tool_calls})")
    print(f"  Wallclock total:   {total_wallclock:.0f}s ({total_wallclock/60:.1f}m)")


if __name__ == "__main__":
    main()
