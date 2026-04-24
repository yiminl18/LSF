"""
Answer generation module: generate natural-language answers from top-k nodes.
"""

from dataclasses import dataclass

from core.pipeline.e2e_utils.cache import CachedLLMCaller
from core.llm.cost import compute_cost

# Prompt template
_PROMPT_TEMPLATE = """\
Answer the question using ONLY the provided context sections from a document.
If the answer cannot be found in the context, respond with "Information not found."

Context sections:
---
{sections}
---

Question: {question}
Answer:"""


@dataclass
class GenerationResult:
    """Answer generation result."""

    answer: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    cache_hit: bool


def build_sections(top_k_nodes: list[dict]) -> str:
    """Format top-k nodes into a context sections string."""
    parts = []
    for i, node in enumerate(top_k_nodes):
        parts.append(
            f"[Section {i + 1}: {node['path_text']}]\n"
            f"{node['text']}\n"
            f"{node['text_span']}"
        )
    return "\n---\n".join(parts)


def generate_answer(
    question: str,
    top_k_nodes: list[dict],
    cached_caller: CachedLLMCaller,
    llm_provider: str = "azure",
    *,
    llm_model: str,
    max_tokens: int = 500,
) -> GenerationResult:
    """
    Generate an answer for a question using top-k retrieved nodes.

    Args:
        question: user question
        top_k_nodes: list of retrieved nodes, each with idx/score/text/text_span/path_text
        cached_caller: LLM caller with caching
        llm_provider: LLM provider (azure/openai/openrouter)
        llm_model: LLM model identifier
        max_tokens: maximum output tokens

    Returns:
        GenerationResult containing answer, token counts, latency, cost, and cache-hit flag.
    """
    sections = build_sections(top_k_nodes)
    prompt = _PROMPT_TEMPLATE.format(sections=sections, question=question)

    call_kwargs = {
        "llm_provider": llm_provider,
        "max_tokens": max_tokens,
        "model": llm_model,
    }
    cache_result = cached_caller.call(prompt, **call_kwargs)

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
