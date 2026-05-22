"""Token-based pricing for LLM calls made inside baselines.

Self-contained: no dependency on a project-wide config module. Mirrors
``chiyu-dev``'s ``core/llm/cost.py`` shape so the per-call sidecar produced by
``mdocagent/openai_model.py`` reports costs comparable to the chiyu-dev
baseline summaries.
"""

from __future__ import annotations

GPT_PRICE_PER_MILLION_INPUT = 2.5
GPT_PRICE_PER_MILLION_OUTPUT = 10.0
GPT_54_PRICE_PER_MILLION_INPUT = 2.5
GPT_54_PRICE_PER_MILLION_CACHED_INPUT = 0.25
GPT_54_PRICE_PER_MILLION_OUTPUT = 15.0
GPT_54_MINI_PRICE_PER_MILLION_INPUT = 0.75
GPT_54_MINI_PRICE_PER_MILLION_CACHED_INPUT = 0.075
GPT_54_MINI_PRICE_PER_MILLION_OUTPUT = 4.5

PROVIDER_PRICES: dict[str, tuple[float, float]] = {
    "azure": (GPT_PRICE_PER_MILLION_INPUT, GPT_PRICE_PER_MILLION_OUTPUT),
}

OPENROUTER_MODEL_PRICES: dict[str, tuple[float, float]] = {
    "z-ai/glm-5.1": (0.95, 3.15),
}

MODEL_PREFIX_PRICES: tuple[tuple[str, tuple[float, float]], ...] = (
    ("openai/gpt-5.4-mini", (GPT_54_MINI_PRICE_PER_MILLION_INPUT, GPT_54_MINI_PRICE_PER_MILLION_OUTPUT)),
    ("gpt-5.4-mini",        (GPT_54_MINI_PRICE_PER_MILLION_INPUT, GPT_54_MINI_PRICE_PER_MILLION_OUTPUT)),
    ("openai/gpt-5.4",      (GPT_54_PRICE_PER_MILLION_INPUT, GPT_54_PRICE_PER_MILLION_OUTPUT)),
    ("gpt-5.4",             (GPT_54_PRICE_PER_MILLION_INPUT, GPT_54_PRICE_PER_MILLION_OUTPUT)),
)

MINI_PRICE_INPUT = 0.15
MINI_PRICE_OUTPUT = 0.60


def _normalize_model_name(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("LLM model must be specified for cost calculation")
    normalized = model.strip().lower()
    if normalized.startswith("openrouter:"):
        return normalized.split(":", 1)[1]
    return normalized


def _match_prefix_price(normalized_model: str) -> tuple[float, float] | None:
    for prefix, prices in MODEL_PREFIX_PRICES:
        if normalized_model == prefix or normalized_model.startswith(f"{prefix}-"):
            return prices
    return None


def get_prices(llm_provider: str, model: str) -> tuple[float, float]:
    normalized = _normalize_model_name(model)
    prefix_match = _match_prefix_price(normalized)
    if prefix_match is not None:
        return prefix_match

    if "mini" in normalized:
        return MINI_PRICE_INPUT, MINI_PRICE_OUTPUT

    provider = (llm_provider or "").strip().lower()
    if provider == "openrouter":
        if normalized in OPENROUTER_MODEL_PRICES:
            return OPENROUTER_MODEL_PRICES[normalized]
        raise ValueError(f"OpenRouter pricing not configured for model={model!r}")

    if provider not in PROVIDER_PRICES:
        raise ValueError(f"Unknown llm_provider: {llm_provider!r}")
    return PROVIDER_PRICES[provider]


def compute_cost(
    input_tokens: int,
    output_tokens: int,
    llm_provider: str,
    model: str,
) -> float:
    """Return USD cost for one chat-completion call."""
    if input_tokens == 0 and output_tokens == 0:
        return 0.0
    p_in, p_out = get_prices(llm_provider, model=model)
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000
