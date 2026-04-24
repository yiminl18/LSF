"""e2e_utils: end-to-end evaluation helpers.

Provides LLM-call caching, cost accounting, answer generation, and judging.
"""

from core.pipeline.e2e_utils.cache import (
    CachedLLMCaller,
    CacheResult,
    DEFAULT_CACHE_DB_PATH,
)
from core.llm.cost import compute_cost
from core.pipeline.e2e_utils.generation import generate_answer, GenerationResult
from core.pipeline.e2e_utils.judge import judge_answer, normalize_ground_truth, JudgeResult
from core.pipeline.e2e_utils.results import build_query_result, build_summary

__all__ = [
    "CachedLLMCaller",
    "CacheResult",
    "DEFAULT_CACHE_DB_PATH",
    "compute_cost",
    "generate_answer",
    "GenerationResult",
    "judge_answer",
    "normalize_ground_truth",
    "JudgeResult",
    "build_query_result",
    "build_summary",
]
