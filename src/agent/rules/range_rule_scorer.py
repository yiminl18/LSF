"""Thin judge/match adapter for retrieved markdown subsets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.pipeline.e2e_utils.cache import CachedLLMCaller
from core.llm.cost import compute_cost
from core.pipeline.e2e_utils.generation import GenerationResult
from core.pipeline.e2e_utils.judge import judge_answer, normalize_ground_truth

JUDGE_METHOD_NAME = "llm_generate_then_llm_judge"
_DEFAULT_ANSWER_PROMPT = """\
Answer the question using ONLY the provided context.
If the answer cannot be found in the context, respond with "Information not found."

Context:
---
{context}
---

Question: {question}
Answer:"""


@dataclass(slots=True, frozen=True)
class RangeRuleScore:
    generated_answer: str
    judge_method: str
    judge_result: bool
    accuracy: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_answer_text(answer: Any) -> str:
    """Normalize any answer payload to a plain string."""
    if answer is None:
        return ""
    if isinstance(answer, str):
        return answer
    if isinstance(answer, list):
        if not answer:
            return ""
        if all(isinstance(item, dict) for item in answer):
            return "\n".join(
                ", ".join(f"{key}: {value}" for key, value in item.items())
                for item in answer
            )
        return "\n".join(normalize_answer_text(item) for item in answer)
    return str(answer)


def _build_answer_prompt(question: str, context: str) -> str:
    return _DEFAULT_ANSWER_PROMPT.replace("{context}", context).replace(
        "{question}", question
    )


def _generate_with_default_prompt(
    question: str,
    context: str,
    cached_caller: CachedLLMCaller,
    llm_provider: str = "azure",
    *,
    llm_model: str,
    max_tokens: int = 500,
) -> GenerationResult:
    """Generate an answer using the built-in scorer prompt."""
    prompt = _build_answer_prompt(question, context)
    call_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "llm_provider": llm_provider,
        "max_tokens": max_tokens,
        "model": llm_model,
    }
    cache_result = cached_caller.call(**call_kwargs)
    cost_usd = compute_cost(
        cache_result.input_tokens,
        cache_result.output_tokens,
        llm_provider,
        model=llm_model,
    )
    return GenerationResult(
        answer=cache_result.response,
        input_tokens=cache_result.input_tokens,
        output_tokens=cache_result.output_tokens,
        latency_ms=cache_result.latency_ms,
        cost_usd=cost_usd,
        cache_hit=cache_result.cache_hit,
    )


def generate_answer_from_text(
    question: str,
    retrieved_text: str,
    cached_caller: CachedLLMCaller,
    llm_provider: str = "azure",
    *,
    llm_model: str,
    max_tokens: int = 500,
) -> GenerationResult:
    """Public generation entry point, reusable by tests and other callers."""
    return _generate_with_default_prompt(
        question=question,
        context=retrieved_text,
        cached_caller=cached_caller,
        llm_provider=llm_provider,
        llm_model=llm_model,
        max_tokens=max_tokens,
    )


def score_generated_answer(
    question: str,
    generated_answer: str,
    ground_truth: str,
    cached_caller: CachedLLMCaller,
    llm_provider: str = "azure",
    *,
    llm_model: str,
    judge_max_tokens: int = 50,
) -> RangeRuleScore:
    """Judge a pre-generated answer directly."""
    normalized_gt = normalize_ground_truth(ground_truth)
    judge = judge_answer(
        question=question,
        generated_answer=generated_answer,
        ground_truth=normalized_gt,
        cached_caller=cached_caller,
        llm_provider=llm_provider,
        llm_model=llm_model,
        max_tokens=judge_max_tokens,
    )
    return RangeRuleScore(
        generated_answer=generated_answer,
        judge_method=JUDGE_METHOD_NAME,
        judge_result=judge.is_correct,
        accuracy=1.0 if judge.is_correct else 0.0,
        metadata={
            "generation": {
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": 0.0,
                "cost_usd": 0.0,
                "cache_hit": True,
            },
            "judge": {
                "input_tokens": judge.input_tokens,
                "output_tokens": judge.output_tokens,
                "latency_ms": judge.latency_ms,
                "cost_usd": judge.cost_usd,
                "cache_hit": judge.cache_hit,
                "raw_response": judge.raw_response,
            },
        },
    )


def score_retrieved_subset(
    question: str,
    retrieved_text: str,
    ground_truth: str,
    cached_caller: CachedLLMCaller,
    llm_provider: str = "azure",
    *,
    llm_model: str,
    gen_max_tokens: int = 500,
    judge_max_tokens: int = 50,
) -> RangeRuleScore:
    """Score a retrieved text subset via generate → judge."""
    generation = generate_answer_from_text(
        question=question,
        retrieved_text=retrieved_text,
        cached_caller=cached_caller,
        llm_provider=llm_provider,
        llm_model=llm_model,
        max_tokens=gen_max_tokens,
    )
    normalized_gt = normalize_ground_truth(ground_truth)
    judge = judge_answer(
        question=question,
        generated_answer=generation.answer,
        ground_truth=normalized_gt,
        cached_caller=cached_caller,
        llm_provider=llm_provider,
        llm_model=llm_model,
        max_tokens=judge_max_tokens,
    )
    return RangeRuleScore(
        generated_answer=generation.answer,
        judge_method=JUDGE_METHOD_NAME,
        judge_result=judge.is_correct,
        accuracy=1.0 if judge.is_correct else 0.0,
        metadata={
            "generation": {
                "input_tokens": generation.input_tokens,
                "output_tokens": generation.output_tokens,
                "latency_ms": generation.latency_ms,
                "cost_usd": generation.cost_usd,
                "cache_hit": generation.cache_hit,
            },
            "judge": {
                "input_tokens": judge.input_tokens,
                "output_tokens": judge.output_tokens,
                "latency_ms": judge.latency_ms,
                "cost_usd": judge.cost_usd,
                "cache_hit": judge.cache_hit,
                "raw_response": judge.raw_response,
            },
        },
    )


__all__ = [
    "JUDGE_METHOD_NAME",
    "RangeRuleScore",
    "generate_answer_from_text",
    "normalize_answer_text",
    "score_generated_answer",
    "score_retrieved_subset",
]
