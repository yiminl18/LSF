#!/usr/bin/env python3
"""Persist a new rule .py file. Free.

Validates that the file defines exactly one `def rule_<name>(doc: dict) -> list[dict]`
matching the requested name, smoke-imports it, and writes it to
<rules-dir>/<question-slug>/<rule_name>.py.

Usage:
    python tools/write_rule.py --question-slug <slug> --name <rule_name> --code-file <path>
    python tools/write_rule.py --question-slug <slug> --name <rule_name> --code "<inline>"
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
from _paths import RULES_BASE_DIR  # noqa: E402


def _validate_source(source: str, rule_name: str) -> tuple[bool, str]:
    """Return (ok, message). Validates AST + function signature."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, f"syntax error: {e}"

    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("rule_")]
    if not funcs:
        return False, "no rule_* function found"
    if not any(f.name == rule_name for f in funcs):
        names = [f.name for f in funcs]
        return False, f"requested name {rule_name!r} not in defined functions {names}"

    target = next(f for f in funcs if f.name == rule_name)
    args_ok = len(target.args.args) >= 1 and target.args.args[0].arg == "doc"
    if not args_ok:
        return False, f"first argument of {rule_name} must be `doc`"
    return True, "ok"


def _smoke_import(py_path: Path, rule_name: str) -> tuple[bool, str]:
    """Try importing the module and resolving the function. Catches NameError/ImportError."""
    try:
        spec = importlib.util.spec_from_file_location(py_path.stem, py_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
    except Exception as e:
        return False, f"import error: {e}"
    fn = getattr(mod, rule_name, None)
    if fn is None:
        return False, f"function {rule_name} not exported"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser(description="Validate + persist a generated rule file.")
    ap.add_argument("--question-slug", required=True)
    ap.add_argument("--name", required=True, help="rule function name, must start with rule_")
    ap.add_argument("--code-file", help="path to a file containing the rule source")
    ap.add_argument("--code", help="inline rule source (use --code-file for multi-line)")
    ap.add_argument("--rules-dir", default=str(RULES_BASE_DIR))
    ap.add_argument("--overwrite", action="store_true", help="overwrite if file already exists")
    ap.add_argument("--format", choices=("text", "json"), default="json")
    args = ap.parse_args()

    if not args.name.startswith("rule_"):
        print(json.dumps({"ok": False, "error": "name must start with rule_"}, ensure_ascii=False))
        sys.exit(2)

    if args.code_file and args.code:
        print(json.dumps({"ok": False, "error": "pass only one of --code-file or --code"}, ensure_ascii=False))
        sys.exit(2)
    if not args.code_file and not args.code:
        print(json.dumps({"ok": False, "error": "must pass --code-file or --code"}, ensure_ascii=False))
        sys.exit(2)

    source = Path(args.code_file).read_text(encoding="utf-8") if args.code_file else (args.code or "")

    ok, msg = _validate_source(source, args.name)
    if not ok:
        print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))
        sys.exit(3)

    folder = Path(args.rules_dir) / args.question_slug
    folder.mkdir(parents=True, exist_ok=True)
    py_path = folder / f"{args.name}.py"
    if py_path.exists() and not args.overwrite:
        print(json.dumps({
            "ok": False,
            "error": f"{py_path} already exists; pass --overwrite to replace",
            "rule_path": str(py_path),
        }, ensure_ascii=False))
        sys.exit(4)

    py_path.write_text(source, encoding="utf-8")

    ok, msg = _smoke_import(py_path, args.name)
    if not ok:
        # Persist the file but report the import failure so the agent can fix it
        result = {"ok": False, "error": msg, "rule_path": str(py_path)}
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(f"FAIL {py_path}: {msg}")
        sys.exit(5)

    result = {
        "ok":         True,
        "rule_name":  args.name,
        "rule_path":  str(py_path),
        "question_slug": args.question_slug,
        "size_bytes": py_path.stat().st_size,
    }
    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"OK {py_path} ({result['size_bytes']} bytes)")


if __name__ == "__main__":
    main()
