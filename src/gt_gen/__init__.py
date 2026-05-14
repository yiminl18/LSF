"""Ground-truth generation utilities."""

from gt_gen.generator import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_INPUT_MODE,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MODEL,
    DocRunResult,
    GenerationSummary,
    QuerySpec,
    generate_ground_truth,
)

__all__ = [
    "DEFAULT_INPUT_MODE",
    "DEFAULT_LLM_PROVIDER",
    "DEFAULT_MODEL",
    "DEFAULT_CLAUDE_MODEL",
    "DocRunResult",
    "GenerationSummary",
    "QuerySpec",
    "generate_ground_truth",
]
