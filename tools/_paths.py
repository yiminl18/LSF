"""Shared path constants for the agentic tools.

All tools read/write under the FinanceBench single-cluster gpt54/one_shot tree.
Override with CLI flags when needed for other variants.
"""

from __future__ import annotations
from pathlib import Path

# Repo root (the LSF directory)
ROOT = Path(__file__).resolve().parents[1]

# Where rule files live
RULES_BASE_DIR = ROOT / "rules/financebench_single_cluster/llm/gpt54/one_shot"

# Where ground-truth labels live
SAMPLED_LABELS_FILE = ROOT / "data/financebench/sample_doc_labels.json"

# Where doc JSONs live
PROCESSING_DIR = ROOT / "data/financebench/processing"

# Where computed costs are cached
COST_PROFILE_DIR = ROOT / "results/financebench_single_cluster/llm/gpt54/one_shot/cost_profile"

# Where per-rule eval (coverage) lives
EVAL_INDIVIDUAL_DIR = ROOT / "results/financebench_single_cluster/llm/gpt54/one_shot/eval_individual"

# Where full-set merge eval lives (defines D*)
EVAL_MERGE_DIR = ROOT / "results/financebench_single_cluster/llm/gpt54/one_shot/eval_merge"

# Where the agentic selector writes its outputs
SELECTED_RULES_AGENT_DIR = ROOT / "results/financebench_single_cluster/llm/gpt54/one_shot/selected_rules_agent"

# Where per-call merge-eval JSON intermediates land (segregated)
SELECTOR_RUN_AGENT_DIR = ROOT / "results/financebench_single_cluster/llm/gpt54/one_shot/selector_run_agent"

# Where per-session traces land
AGENT_TRACE_DIR = ROOT / "results/financebench_single_cluster/llm/gpt54/one_shot/agent_trace"
