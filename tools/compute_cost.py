#!/usr/bin/env python3
"""Compute avg_cost_ratio for a list of rules on the sampled docs. No LLM.

Reads (or builds + caches) the cost profile via
`src/rule_refinement/cost_profile.py::load_or_compute_cost_profile`.

Usage:
    python tools/compute_cost.py --question-slug <slug> \
        --rules rule_a rule_b rule_c
    # Or compute for the entire rule pool:
    python tools/compute_cost.py --question-slug <slug> --all
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

from rule_refinement.cost_profile import load_or_compute_cost_profile  # noqa: E402
from _paths import (                                                   # noqa: E402
    RULES_BASE_DIR, SAMPLED_LABELS_FILE, PROCESSING_DIR, COST_PROFILE_DIR,
)


def main():
    ap = argparse.ArgumentParser(description="Compute avg_cost_ratio of a rule subset.")
    ap.add_argument("--question-slug", required=True)
    ap.add_argument("--rules", nargs="*", default=[], help="rule_<name> names (no .py)")
    ap.add_argument("--all", action="store_true", help="use every rule in the folder")
    ap.add_argument("--rules-dir",   default=str(RULES_BASE_DIR))
    ap.add_argument("--labels-file", default=str(SAMPLED_LABELS_FILE))
    ap.add_argument("--processing-dir", default=str(PROCESSING_DIR))
    ap.add_argument("--cache-dir", default=str(COST_PROFILE_DIR))
    ap.add_argument("--format", choices=("text", "json"), default="json")
    args = ap.parse_args()

    folder = Path(args.rules_dir) / args.question_slug
    if not folder.is_dir():
        print(f"ERROR: rule folder not found: {folder}", file=sys.stderr)
        sys.exit(2)

    # Load doc paths from labels file (sampled docs only)
    labels = json.loads(Path(args.labels_file).read_text(encoding="utf-8"))
    doc_paths = []
    for pdf_key in labels:
        dn = pdf_key.replace(".pdf", "")
        path = Path(args.processing_dir) / f"{dn}_reconstructed.json"
        if path.exists():
            doc_paths.append(path)

    cache_path = Path(args.cache_dir) / f"{args.question_slug}.json"
    profile = load_or_compute_cost_profile(
        rules_dir=str(folder), doc_paths=doc_paths, cache_path=cache_path,
    )

    if args.all:
        rules = sorted(profile.keys())
    else:
        rules = args.rules

    per_rule = {}
    for r in rules:
        per_rule[r] = profile.get(r, {}).get("avg_cost_ratio", None)

    valid_vals = [v for v in per_rule.values() if v is not None]
    result = {
        "question_slug":       args.question_slug,
        "n_rules":             len(rules),
        "per_rule":            {r: (round(v, 6) if v is not None else None) for r, v in per_rule.items()},
        "sum_avg_cost_ratio":  round(sum(valid_vals), 6) if valid_vals else 0.0,
        "max_avg_cost_ratio":  round(max(valid_vals), 6) if valid_vals else 0.0,
        "cache_path":          str(cache_path),
        "missing_in_profile":  [r for r in rules if per_rule.get(r) is None],
    }

    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"question: {args.question_slug}")
        print(f"rules: {len(rules)}")
        for r, v in per_rule.items():
            print(f"  {r}: {v}")
        print(f"sum = {result['sum_avg_cost_ratio']}")
        print(f"max = {result['max_avg_cost_ratio']}")
        if result["missing_in_profile"]:
            print(f"missing: {result['missing_in_profile']}")


if __name__ == "__main__":
    main()
