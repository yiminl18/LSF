"""
LLM call cost calculation.

Centralized token-based API pricing for all supported providers.
"""

from core.config import (
    GPT_PRICE_PER_MILLION_INPUT,
    GPT_PRICE_PER_MILLION_OUTPUT,
    GPT_54_MINI_PRICE_PER_MILLION_CACHED_INPUT,
    GPT_54_MINI_PRICE_PER_MILLION_INPUT,
    GPT_54_MINI_PRICE_PER_MILLION_OUTPUT,
    GPT_54_PRICE_PER_MILLION_CACHED_INPUT,
    GPT_54_PRICE_PER_MILLION_INPUT,
    GPT_54_PRICE_PER_MILLION_OUTPUT,
)

# provider -> (input_price, output_price) per million tokens
PROVIDER_PRICES: dict[str, tuple[float, float]] = {
    "azure": (GPT_PRICE_PER_MILLION_INPUT, GPT_PRICE_PER_MILLION_OUTPUT),
}

# OpenRouter model-specific approximate pricing (per million tokens)
OPENROUTER_MODEL_PRICES: dict[str, tuple[float, float]] = {
    "z-ai/glm-5.1": (0.95, 3.15),
}

MODEL_PREFIX_PRICES: tuple[tuple[str, tuple[float, float]], ...] = (
    (
        "openai/gpt-5.4-mini",
        (
            GPT_54_MINI_PRICE_PER_MILLION_INPUT,
            GPT_54_MINI_PRICE_PER_MILLION_OUTPUT,
        ),
    ),
    (
        "gpt-5.4-mini",
        (
            GPT_54_MINI_PRICE_PER_MILLION_INPUT,
            GPT_54_MINI_PRICE_PER_MILLION_OUTPUT,
        ),
    ),
    (
        "openai/gpt-5.4",
        (GPT_54_PRICE_PER_MILLION_INPUT, GPT_54_PRICE_PER_MILLION_OUTPUT),
    ),
    ("gpt-5.4", (GPT_54_PRICE_PER_MILLION_INPUT, GPT_54_PRICE_PER_MILLION_OUTPUT)),
)

MODEL_PREFIX_CACHED_INPUT_PRICES: tuple[tuple[str, float], ...] = (
    ("openai/gpt-5.4-mini", GPT_54_MINI_PRICE_PER_MILLION_CACHED_INPUT),
    ("gpt-5.4-mini", GPT_54_MINI_PRICE_PER_MILLION_CACHED_INPUT),
    ("openai/gpt-5.4", GPT_54_PRICE_PER_MILLION_CACHED_INPUT),
    ("gpt-5.4", GPT_54_PRICE_PER_MILLION_CACHED_INPUT),
)

# Generic mini-model pricing shared across providers.
MINI_PRICE_INPUT = 0.15
MINI_PRICE_CACHED_INPUT = 0.075
MINI_PRICE_OUTPUT = 0.60


def compute_cost(
    input_tokens: int,
    output_tokens: int,
    llm_provider: str,
    model: str,
) -> float:
    """Compute LLM call cost in USD.

    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        llm_provider: Provider name, such as azure or openrouter.
        model: Model name for provider-specific pricing.

    Returns:
        Cost in USD.

    Raises:
        ValueError: Unknown provider or missing model pricing.
    """
    if input_tokens == 0 and output_tokens == 0:
        return 0.0

    price_input, price_output = get_prices(llm_provider, model=model)
    return (input_tokens * price_input + output_tokens * price_output) / 1_000_000


def compute_cost_with_cached_input(
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    llm_provider: str,
    model: str,
) -> float:
    """Compute LLM call cost in USD with cached-input pricing.

    API usage typically reports cached input tokens inside total input tokens,
    so cached tokens are split out and billed at the cached-input rate.
    """
    if input_tokens == 0 and cached_input_tokens == 0 and output_tokens == 0:
        return 0.0

    input_price, cached_input_price, output_price = get_prices_with_cached_input(
        llm_provider, model=model
    )
    cached_tokens = max(0, min(cached_input_tokens, input_tokens))
    uncached_tokens = max(0, input_tokens - cached_tokens)
    return (
        uncached_tokens * input_price
        + cached_tokens * cached_input_price
        + output_tokens * output_price
    ) / 1_000_000


def _normalize_model_name(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("LLM model must be specified for cost calculation")
    normalized_model = model.strip().lower()
    if normalized_model == "unspec" + "ified":
        raise ValueError("LLM model uses a reserved invalid name")
    if normalized_model.startswith("openrouter:"):
        return normalized_model.split(":", 1)[1]
    return normalized_model


def _match_model_prefix_price(normalized_model: str) -> tuple[float, float] | None:
    for prefix, prices in MODEL_PREFIX_PRICES:
        if normalized_model == prefix or normalized_model.startswith(f"{prefix}-"):
            return prices
    return None


def _match_model_prefix_cached_input_price(normalized_model: str) -> float | None:
    for prefix, price in MODEL_PREFIX_CACHED_INPUT_PRICES:
        if normalized_model == prefix or normalized_model.startswith(f"{prefix}-"):
            return price
    return None


def get_prices(llm_provider: str, model: str) -> tuple[float, float]:
    """Return (input_price, output_price) per million tokens.

    Args:
        llm_provider: Provider name.
        model: Model name. Generic mini pricing is used for mini models
            without model-specific pricing.

    Returns:
        (input_price_per_million, output_price_per_million)
    """
    normalized_model = _normalize_model_name(model)
    model_specific_prices = _match_model_prefix_price(normalized_model)
    if model_specific_prices is not None:
        return model_specific_prices

    if "mini" in normalized_model:
        return MINI_PRICE_INPUT, MINI_PRICE_OUTPUT

    if llm_provider == "openrouter":
        if normalized_model in OPENROUTER_MODEL_PRICES:
            return OPENROUTER_MODEL_PRICES[normalized_model]
        raise ValueError(
            f"OpenRouter model pricing is not configured for model={model!r}"
        )

    if llm_provider not in PROVIDER_PRICES:
        raise ValueError(f"Unknown llm_provider: {llm_provider!r}")

    return PROVIDER_PRICES[llm_provider]


def get_prices_with_cached_input(
    llm_provider: str, model: str
) -> tuple[float, float, float]:
    """Return (input, cached_input, output) prices per million tokens."""
    normalized_model = _normalize_model_name(model)
    input_price, output_price = get_prices(llm_provider, model=model)
    normalized_provider = str(llm_provider).strip().lower()
    cached_input_price = (
        _match_model_prefix_cached_input_price(normalized_model)
        if normalized_provider == "azure"
        else None
    )
    if cached_input_price is not None:
        return input_price, cached_input_price, output_price
    if normalized_provider == "azure" and "mini" in normalized_model and (
        input_price,
        output_price,
    ) == (
        MINI_PRICE_INPUT,
        MINI_PRICE_OUTPUT,
    ):
        return input_price, MINI_PRICE_CACHED_INPUT, output_price
    return input_price, input_price, output_price
