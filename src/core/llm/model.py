"""
Unified Model API Wrapper

Provides a unified model call interface that routes to the appropriate
provider-specific implementation.

Main functions:
- llm_call(): Unified LLM call, routes by provider
- reset_llm_cost() / get_llm_cost(): Per-provider cost counter management
- model(): Legacy interface (kept for compatibility)
- test_model(): Test whether a model is working
"""

from typing import Set

from core.llm.gpt_4o_azure import gpt_4o_azure
from core.llm.gpt_4o_azure import reset_cost_counter as _azure_reset
from core.llm.gpt_4o_azure import get_cumulative_cost as _azure_get_cost
from core.llm.gpt_4o_openrouter import gpt_4o_openrouter
from core.llm.gpt_4o_openrouter import reset_cost_counter as _openrouter_reset
from core.llm.gpt_4o_openrouter import get_cumulative_cost as _openrouter_get_cost
from core.llm.gpt_4o_openai import gpt_4o_openai
from core.llm.gpt_4o_openai import reset_cost_counter as _openai_reset
from core.llm.gpt_4o_openai import get_cumulative_cost as _openai_get_cost

LLM_PROVIDERS: Set[str] = {"azure", "openai", "openrouter"}


def _validate_provider(llm_provider: str) -> None:
    if llm_provider not in LLM_PROVIDERS:
        raise ValueError(
            f"Unknown llm_provider={llm_provider!r}, expected one of {sorted(LLM_PROVIDERS)}"
        )


def llm_call(
    prompt: str, llm_provider: str = "azure", max_tokens: int = 800, **kwargs
) -> str:
    """
    Unified LLM call that routes to the appropriate provider implementation.

    Args:
        prompt: Prompt text
        llm_provider: "azure", "openai", or "openrouter"
        max_tokens: Max output token count
        **kwargs: Extra arguments forwarded to the provider (e.g. temperature, estimate_cost)

    Returns:
        Model response text
    """
    _validate_provider(llm_provider)
    if llm_provider == "azure":
        return gpt_4o_azure(prompt, max_tokens=max_tokens, **kwargs)
    if llm_provider == "openrouter":
        return gpt_4o_openrouter(prompt, max_tokens=max_tokens, **kwargs)
    return gpt_4o_openai(prompt, max_tokens=max_tokens, **kwargs)


def reset_llm_cost(llm_provider: str = "azure") -> None:
    """Reset the cost counter for the specified provider."""
    _validate_provider(llm_provider)
    if llm_provider == "azure":
        _azure_reset()
    elif llm_provider == "openrouter":
        _openrouter_reset()
    else:
        _openai_reset()


def get_llm_cost(llm_provider: str = "azure") -> float:
    """Get the cumulative cost for the specified provider."""
    _validate_provider(llm_provider)
    if llm_provider == "azure":
        return _azure_get_cost()
    if llm_provider == "openrouter":
        return _openrouter_get_cost()
    return _openai_get_cost()
