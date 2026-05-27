"""CLI tools exposed to the agentic rule selector via Bash.

Each script wraps a primitive from src/rule_refine/selection/ behind a thin CLI:

  - compute_cost.py       → cost_profile.load_or_compute_cost_profile
  - compute_coverage.py   → coverage_check.load_or_compute_coverage
  - verify_accuracy.py    → rule_apply_merge + eval_judge.judge
  - list_rules.py         → scans rules/<slug>/ and parses docstrings
  - inspect_rule.py       → prints a single rule's source

See docs/rule_selection_agentic.md for the design rationale.
"""
