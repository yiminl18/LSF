#!/usr/bin/env python3
"""List all rules in the pool for a question with one-line docstrings.

Lets the agent reason about rules semantically before paying for verification.
Wraps a directory scan + AST docstring parse — no LLM calls.

Usage:
    python tools/list_rules.py --question-slug what_is_the_registrants_telephone_number_10_llm
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
from _paths import RULES_BASE_DIR


def _docstring_of_first_rule_function(py_path: Path) -> str:
    """Parse the .py file with ast and return the first function's docstring (one line)."""
    try:
        tree = ast.parse(py_path.read_text(encoding="utf-8"))
    except Exception as e:
        return f"<parse error: {e}>"
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("rule_"):
            doc = ast.get_docstring(node) or ""
            # Collapse to single line
            return " ".join(doc.split())
    return "<no rule_ function found>"


def main():
    ap = argparse.ArgumentParser(description="List rules with one-line docstrings.")
    ap.add_argument("--question-slug", required=True,
                    help="e.g. what_is_the_registrants_telephone_number_10_llm")
    ap.add_argument("--rules-dir", default=str(RULES_BASE_DIR),
                    help=f"base rules dir (default: {RULES_BASE_DIR})")
    ap.add_argument("--format", choices=("text", "json"), default="text")
    args = ap.parse_args()

    folder = Path(args.rules_dir) / args.question_slug
    if not folder.is_dir():
        print(f"ERROR: rule folder not found: {folder}", file=sys.stderr)
        sys.exit(2)

    rules: list[dict] = []
    for py in sorted(folder.glob("rule_*.py")):
        rules.append({
            "rule_name":   py.stem,
            "description": _docstring_of_first_rule_function(py),
            "path":        str(py.relative_to(Path.cwd())) if py.is_absolute() else str(py),
        })

    if args.format == "json":
        print(json.dumps({"question_slug": args.question_slug, "n": len(rules), "rules": rules},
                         indent=2, ensure_ascii=False))
    else:
        print(f"# {len(rules)} rules in {folder}")
        for r in rules:
            print(f"- {r['rule_name']}: {r['description']}")


if __name__ == "__main__":
    main()
