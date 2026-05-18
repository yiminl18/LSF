"""Ground-truth generation utilities."""

from gt_gen.generator import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_INPUT_MODE,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MODEL,
    BatchGenerationSummary,
    DocRunResult,
    GenerationSummary,
    QuerySpec,
    generate_ground_truth,
    generate_ground_truth_for_queries,
)

__all__ = [
    "DEFAULT_INPUT_MODE",
    "DEFAULT_LLM_PROVIDER",
    "DEFAULT_MODEL",
    "DEFAULT_CLAUDE_MODEL",
    "BatchGenerationSummary",
    "DocRunResult",
    "GenerationSummary",
    "QuerySpec",
    "generate_ground_truth",
    "generate_ground_truth_for_queries",
]
