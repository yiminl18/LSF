#!/usr/bin/env python3
"""Look up cov(r) for each rule from cached eval_individual outputs. No LLM.

Reads the existing `_eval.json` files via
`src/rule_refinement/coverage_check.py::load_or_compute_coverage`.

If `eval_individual` has not been run for a rule, cov falls back to 0.0
(the agent can still proceed; it just won't have signal for that rule).

Usage:
    python tools/compute_coverage.py --question-slug <slug> --rules rule_a rule_b
    python tools/compute_coverage.py --question-slug <slug> --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_THIS))

from rule_refinement.coverage_check import load_or_compute_coverage  # noqa: E402
from _paths import RULES_BASE_DIR, EVAL_INDIVIDUAL_DIR               # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Look up cov(r) for a rule subset.")
    ap.add_argument("--question-slug", required=True)
    ap.add_argument("--rules", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rules-dir",   default=str(RULES_BASE_DIR))
    ap.add_argument("--eval-individual-dir", default=str(EVAL_INDIVIDUAL_DIR))
    ap.add_argument("--format", choices=("text", "json"), default="json")
    args = ap.parse_args()

    folder = Path(args.rules_dir) / args.question_slug
    if not folder.is_dir():
        print(f"ERROR: rule folder not found: {folder}", file=sys.stderr)
        sys.exit(2)

    if args.all:
        rules = sorted(f.stem for f in folder.glob("rule_*.py"))
    else:
        rules = args.rules

    eval_dir = Path(args.eval_individual_dir) / args.question_slug

    per_rule = {}
    for r in rules:
        eval_path = eval_dir / f"{r}_eval.json"
        per_rule[r] = load_or_compute_coverage(r, eval_path, lambda: 0.0)

    valid = [v for v in per_rule.values() if v is not None]
    result = {
        "question_slug":  args.question_slug,
        "n_rules":        len(rules),
        "per_rule":       {r: round(v, 4) for r, v in per_rule.items()},
        "min_cov":        round(min(valid), 4) if valid else 0.0,
        "mean_cov":       round(sum(valid)/len(valid), 4) if valid else 0.0,
        "eval_individual_dir": str(eval_dir),
    }

    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"question: {args.question_slug}")
        for r, v in per_rule.items():
            print(f"  {r}: {v}")
        print(f"min={result['min_cov']}  mean={result['mean_cov']}")


if __name__ == "__main__":
    main()
