#!/usr/bin/env python3
"""Print the full source of a single rule file.

Lets the agent read a specific rule to understand its logic before keeping
or dropping it. No LLM calls.

Usage:
    python tools/inspect_rule.py --question-slug <slug> --rule rule_xxx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
from _paths import RULES_BASE_DIR


def main():
    ap = argparse.ArgumentParser(description="Print the source of one rule.")
    ap.add_argument("--question-slug", required=True)
    ap.add_argument("--rule",          required=True, help="rule_<name> (no .py)")
    ap.add_argument("--rules-dir", default=str(RULES_BASE_DIR))
    args = ap.parse_args()

    py = Path(args.rules_dir) / args.question_slug / f"{args.rule}.py"
    if not py.exists():
        print(f"ERROR: rule file not found: {py}", file=sys.stderr)
        sys.exit(2)
    print(py.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
