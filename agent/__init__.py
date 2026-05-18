"""Agentic rule selection — Claude Code as the outer-loop optimiser.

The driver `run_agent_select.py` spawns one `claude -p` session per question,
passing the task prompt from `task_prompt.md`. The agent uses the tools in
`tools/` (Bash invocations) to inspect rules, compute cost/coverage, and
verify accuracy.

See docs/rule_selection_agentic.md for the design and
docs/rule_selection_agentic_report.md for the implementation report.
"""
