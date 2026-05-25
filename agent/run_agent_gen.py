"""Driver: spawn one Claude Code session per question to GENERATE rules from JSON.

Mirrors agent/run_agent_select.py. Difference: the agent's task is to author
new rules (write_rule) rather than to subset an existing pool. The rule folder
starts empty; the agent fills it.

Usage:
    # Task 1: random sample
    python agent/run_agent_gen.py --sample-set random

    # Task 2: FPS sample
    python agent/run_agent_gen.py --sample-set fps

    # Single question, dry-run prompt:
    python agent/run_agent_gen.py --sample-set random --slug what_is_the_registrants_exact_name_10_agentic --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
sys.path.insert(0, str(_ROOT))

TASK_PROMPT_FILE = _THIS / "task_prompt_gen.md"
QUERIES_FILE     = _ROOT / "data/financebench/sample_queries.txt"

_MODEL_ALIASES = {
    "opus":   "claude-opus-4-5",
    "opus47": "claude-opus-4-7",
    "sonnet": "claude-sonnet-4-5",
    "haiku":  "claude-haiku-4-5-20251001",
}


# ── Sample-set config ────────────────────────────────────────────────────────

def sample_set_config(name: str) -> dict:
    """Return paths + slug-suffix for the given sample set."""
    if name == "random":
        return {
            "labels_file":         _ROOT / "data/financebench/sample/single_cluster/random/sample_doc_labels.json",
            "slug_suffix":         "_10_agentic",
            "rules_dir":           _ROOT / "rules/financebench/lsf/single_cluster/agent/opus47/agentic/raw",
            "results_dir":         _ROOT / "results/financebench/lsf/single_cluster/agent/opus47/agentic/raw",
        }
    if name == "fps":
        return {
            "labels_file":         _ROOT / "data/financebench/sample/single_cluster/fps/sample_doc_labels.json",
            "slug_suffix":         "_10_agentic_fps",
            "rules_dir":           _ROOT / "rules/financebench/lsf/single_cluster/agent/opus47/agentic_fps/raw",
            "results_dir":         _ROOT / "results/financebench/lsf/single_cluster/agent/opus47/agentic_fps/raw",
        }
    raise ValueError(f"unknown sample-set: {name!r}")


def make_slug(question: str, suffix: str) -> str:
    s = question.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60] + suffix


def build_prompt(
    *, question: str, question_slug: str, budget: int, model: str, cfg: dict
) -> str:
    template     = TASK_PROMPT_FILE.read_text(encoding="utf-8")
    rules_dir    = cfg["rules_dir"]
    results_dir  = cfg["results_dir"]
    output_path  = results_dir / "selected_rules_gen" / f"{question_slug}.json"
    trace_path   = results_dir / "agent_trace"       / f"{question_slug}.jsonl"
    cost_cache   = results_dir / "cost_profile"
    eval_indiv   = results_dir / "eval_individual"
    selector_run = results_dir / "selector_run_agent"
    processing   = _ROOT / "data/financebench/processing"

    return template.format(
        question=question,
        question_slug=question_slug,
        budget=budget,
        model=model,
        labels_file=str(cfg["labels_file"].relative_to(_ROOT)),
        processing_dir=str(processing.relative_to(_ROOT)),
        rules_dir=str(rules_dir.relative_to(_ROOT)),
        output_path=str(output_path.relative_to(_ROOT)),
        trace_path=str(trace_path.relative_to(_ROOT)),
        cost_cache_dir=str(cost_cache.relative_to(_ROOT)),
        eval_individual_dir=str(eval_indiv.relative_to(_ROOT)),
        selector_run_dir=str(selector_run.relative_to(_ROOT)),
    )


def run_agent_for_question(
    *, question: str, question_slug: str, budget: int, model: str,
    cfg: dict, timeout: int = 3600, dry_run: bool = False,
) -> dict:
    prompt = build_prompt(
        question=question, question_slug=question_slug,
        budget=budget, model=model, cfg=cfg,
    )
    resolved_model = _MODEL_ALIASES.get(model, model)

    if dry_run:
        return {
            "status":        "dry_run",
            "question":      question,
            "question_slug": question_slug,
            "prompt_chars":  len(prompt),
            "model":         resolved_model,
        }

    # Make sure all output dirs exist before we hand off to claude
    (cfg["rules_dir"]   / question_slug).mkdir(parents=True, exist_ok=True)
    (cfg["results_dir"] / "selected_rules_gen").mkdir(parents=True, exist_ok=True)
    (cfg["results_dir"] / "agent_trace").mkdir(parents=True, exist_ok=True)
    (cfg["results_dir"] / "cost_profile").mkdir(parents=True, exist_ok=True)
    (cfg["results_dir"] / "eval_individual").mkdir(parents=True, exist_ok=True)
    (cfg["results_dir"] / "selector_run_agent").mkdir(parents=True, exist_ok=True)

    cmd = [
        "claude", "--model", resolved_model,
        "--output-format", "json",
        "--dangerously-skip-permissions",
        "-p", prompt,
    ]
    t0 = time.time()
    try:
        res = subprocess.run(
            cmd, capture_output=True, text=True,
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

    opus_usage      = {}
    opus_total_cost = None
    agent_text      = res.stdout or ""
    summary_line    = ""

    try:
        claude_payload  = json.loads(res.stdout or "{}")
        opus_usage      = claude_payload.get("usage", {}) or {}
        opus_total_cost = claude_payload.get("total_cost_usd")
        agent_text      = claude_payload.get("result", "") or ""
    except json.JSONDecodeError:
        pass

    for line in agent_text.splitlines():
        if line.startswith("AGENTIC_GEN_DONE"):
            summary_line = line.strip()
            break

    output_json = cfg["results_dir"] / "selected_rules_gen" / f"{question_slug}.json"
    output_data = None
    if output_json.exists():
        try:
            output_data = json.loads(output_json.read_text(encoding="utf-8"))
        except Exception:
            output_data = None

    return {
        "status":             "ok" if res.returncode == 0 else f"exit_{res.returncode}",
        "question":           question,
        "question_slug":      question_slug,
        "wallclock_seconds":  wallclock,
        "model":              resolved_model,
        "summary_line":       summary_line,
        "opus_usage":         opus_usage,
        "opus_total_cost_usd": opus_total_cost,
        "output_json":        str(output_json),
        "output_data":        output_data,
        "stdout_tail":        agent_text[-1000:],
        "stderr_tail":        (res.stderr or "")[-1000:],
    }


def main():
    ap = argparse.ArgumentParser(description="Spawn Claude sessions to generate rules from scratch.")
    ap.add_argument("--sample-set", choices=("random", "fps"), required=True,
                    help="random = data/financebench/sample/single_cluster/random/sample_doc_labels.json; "
                         "fps = data/financebench/sample/single_cluster/fps/sample_doc_labels.json")
    ap.add_argument("--slug", help="run a single question slug (matches the rule folder name)")
    ap.add_argument("--budget", type=int, default=30,
                    help="max verify_accuracy calls per question (default 30)")
    ap.add_argument("--model", default="opus47",
                    help=f"model alias (default opus47). Options: {', '.join(_MODEL_ALIASES)}")
    ap.add_argument("--dry-run", action="store_true",
                    help="don't actually spawn claude — print the prompt and exit")
    ap.add_argument("--timeout", type=int, default=5400,
                    help="per-question subprocess timeout in seconds (default 5400 = 90m)")
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="skip a question if its selected_rules_gen/<slug>.json already exists (default on)")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    cfg = sample_set_config(args.sample_set)
    questions = [l.strip() for l in QUERIES_FILE.read_text().splitlines() if l.strip()]

    if args.slug:
        questions = [q for q in questions if make_slug(q, cfg["slug_suffix"]) == args.slug]
        if not questions:
            print(f"ERROR: no question maps to slug {args.slug}", file=sys.stderr)
            sys.exit(2)

    print(f"sample_set={args.sample_set}  questions={len(questions)}  model={args.model}  budget={args.budget}")
    print(f"rules_dir   = {cfg['rules_dir']}")
    print(f"results_dir = {cfg['results_dir']}")

    results = []
    for q in questions:
        slug = make_slug(q, cfg["slug_suffix"])
        output_json = cfg["results_dir"] / "selected_rules_gen" / f"{slug}.json"
        if args.skip_existing and output_json.exists():
            print(f"\nSKIP (exists): {slug}")
            try:
                results.append({
                    "status":        "skipped",
                    "question":      q,
                    "question_slug": slug,
                    "output_data":   json.loads(output_json.read_text(encoding="utf-8")),
                })
            except Exception:
                results.append({"status": "skipped", "question": q, "question_slug": slug})
            continue

        print(f"\n{'='*72}\nQuestion: {q}\nSlug:     {slug}")
        out = run_agent_for_question(
            question=q, question_slug=slug,
            budget=args.budget, model=args.model,
            cfg=cfg, timeout=args.timeout, dry_run=args.dry_run,
        )
        results.append(out)
        if args.dry_run:
            print(f"  [dry-run] prompt length = {out['prompt_chars']} chars")
        else:
            print(f"  status={out['status']}  wall={out.get('wallclock_seconds')}s")
            if out.get("summary_line"):
                print(f"  {out['summary_line']}")
            if out.get("output_data"):
                d = out["output_data"]
                print(f"  rules={len(d.get('selected_rules', []))}  "
                      f"match_rate={d.get('match_rate_on_sampled')}  "
                      f"sum_cost={d.get('selected_avg_cost_ratio_sum')}")

    # Driver summary
    print(f"\n{'='*72}\nDriver summary:")
    total_opus_in = total_opus_out = 0
    total_tool_in = total_tool_out = total_tool_calls = 0
    total_wallclock = 0.0
    total_cost_usd  = 0.0
    for r in results:
        d   = r.get("output_data") or {}
        opu = r.get("opus_usage")  or {}
        opus_in_tok  = (opu.get("input_tokens", 0) or 0) + (opu.get("cache_read_input_tokens", 0) or 0)
        opus_out_tok = opu.get("output_tokens", 0) or 0
        total_opus_in    += opus_in_tok
        total_opus_out   += opus_out_tok
        total_tool_in    += d.get("tool_input_tokens", 0) or 0
        total_tool_out   += d.get("tool_output_tokens", 0) or 0
        total_tool_calls += d.get("tool_llm_calls", 0) or 0
        total_wallclock  += r.get("wallclock_seconds", 0) or 0
        if r.get("opus_total_cost_usd") is not None:
            total_cost_usd += r["opus_total_cost_usd"]
        print(f"  {r['question_slug']:<60}  status={r.get('status','?'):<10}  "
              f"rules={len(d.get('selected_rules', [])):>2}  "
              f"match={str(d.get('match_rate_on_sampled', '?')):>5}  "
              f"sum_cost={str(d.get('selected_avg_cost_ratio_sum', '?')):>6}  "
              f"wall={r.get('wallclock_seconds', 0) or 0:>5.0f}s")

    print(f"\n{'='*72}\nTotals across {len(results)} question(s):")
    print(f"  Opus tokens:       in={total_opus_in:,}  out={total_opus_out:,}")
    print(f"  Tool gpt54 tokens: in={total_tool_in:,}  out={total_tool_out:,}  (calls={total_tool_calls})")
    print(f"  Wallclock total:   {total_wallclock:.0f}s ({total_wallclock/60:.1f}m)")
    if total_cost_usd > 0:
        print(f"  Opus cost (reported by CLI): ${total_cost_usd:.4f}")


if __name__ == "__main__":
    main()
