"""
Answer judging module.

Uses an LLM to compare a generated answer against a reference answer and decide correctness.
"""

from dataclasses import dataclass
from typing import Any

from core.pipeline.e2e_utils.cache import CachedLLMCaller, CacheResult
from core.llm.cost import compute_cost

# Judge prompt template
_JUDGE_PROMPT_TEMPLATE = """You are evaluating whether a generated answer is correct by comparing it to a reference answer.

Question: {question}
Reference answer: {ground_truth}
Generated answer: {generated_answer}

Is the generated answer substantially correct? Consider the following:
- Minor wording differences are acceptable.
- The generated answer must capture the key factual content of the reference.
- If the reference is a list, the generated answer should include all key items.

Return ONLY "True" or "False"."""


@dataclass
class JudgeResult:
    """LLM judge result."""

    is_correct: bool
    raw_response: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    cache_hit: bool


def normalize_ground_truth(gt_value: Any) -> str:
    """
    Normalize a ground truth value to a plain string.

    Args:
        gt_value: str, list[str], or list[dict]

    Returns:
        Normalized string.
    """
    if isinstance(gt_value, str):
        return gt_value
    if isinstance(gt_value, list):
        if len(gt_value) == 0:
            return ""
        # list[dict]: format each dict as "key1: value1, key2: value2"
        if isinstance(gt_value[0], dict):
            rows = [
                ", ".join(f"{k}: {v}" for k, v in item.items()) for item in gt_value
            ]
            return "\n".join(rows)
        # list[str]
        return "\n".join(str(item) for item in gt_value)
    return str(gt_value)


def judge_answer(
    question: str,
    generated_answer: str,
    ground_truth: str,
    cached_caller: CachedLLMCaller,
    llm_provider: str = "azure",
    *,
    llm_model: str,
    max_tokens: int = 50,
) -> JudgeResult:
    """
    Call the LLM to determine whether the generated answer substantially matches the reference.

    Args:
        question: original question
        generated_answer: answer produced by the model
        ground_truth: reference answer (already processed by normalize_ground_truth)
        cached_caller: LLM caller with caching
        llm_provider: LLM provider
        llm_model: LLM model identifier
        max_tokens: maximum output tokens

    Returns:
        JudgeResult
    """
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        generated_answer=generated_answer,
    )

    call_kwargs = {
        "prompt": prompt,
        "llm_provider": llm_provider,
        "max_tokens": max_tokens,
        "model": llm_model,
    }
    result: CacheResult = cached_caller.call(**call_kwargs)

    # Parse True/False; default to False on parse failure.
    cleaned = result.response.strip().lower().strip('"').strip("'").strip(".")
    is_correct = cleaned == "true"

    cost_usd = compute_cost(
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        llm_provider=llm_provider,
        model=llm_model,
    )

    return JudgeResult(
        is_correct=is_correct,
        raw_response=result.response,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
        cost_usd=cost_usd,
        cache_hit=result.cache_hit,
    )
