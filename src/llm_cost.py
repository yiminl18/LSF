"""LLM call cost computation (ported from LSF-dev core/llm/cost.py).

Cost in USD from input/output token counts. Prices are per 1M tokens. gpt-5.4 on
Pioneer/OpenAI/Azure share the same list price ($2.50 / $15.00); mini is cheaper.
Model-prefix match wins over provider default so "gpt-5.4-mini" never falls under
the "gpt-5.4" price.
"""
from __future__ import annotations

# per-1M (input, output) USD
GPT_54_PRICE = (2.50, 15.0)
GPT_54_MINI_PRICE = (0.75, 4.5)

# Longest/most-specific prefix first.
_MODEL_PREFIX_PRICES: tuple[tuple[str, tuple[float, float]], ...] = (
    ("gpt-5.4-mini", GPT_54_MINI_PRICE),
    ("gpt-5.4", GPT_54_PRICE),
)

# Fallback by provider when the model id isn't recognized.
_PROVIDER_DEFAULT_PRICE: dict[str, tuple[float, float]] = {
    "pioneer": GPT_54_PRICE,
    "openai": GPT_54_PRICE,
    "azure": GPT_54_PRICE,
}


def get_prices(llm_provider: str, model: str) -> tuple[float, float]:
    """Return (input_price_per_1M, output_price_per_1M) for a model/provider."""
    m = (model or "").strip().lower()
    for prefix, prices in _MODEL_PREFIX_PRICES:
        if m == prefix or m.startswith(prefix + "-") or m.startswith(prefix):
            # exact, hyphen-suffixed, or dated variant (e.g. gpt-5.4-2026-03-05)
            if m == prefix or m.startswith(prefix + "-"):
                return prices
    if "mini" in m:
        return GPT_54_MINI_PRICE
    if "gpt-5.4" in m:
        return GPT_54_PRICE
    default = _PROVIDER_DEFAULT_PRICE.get((llm_provider or "").strip().lower())
    if default is not None:
        return default
    raise ValueError(f"no price for provider={llm_provider!r} model={model!r}")


def compute_cost(input_tokens: int, output_tokens: int, llm_provider: str, model: str) -> float:
    """USD cost for one (or aggregated) call(s)."""
    if not input_tokens and not output_tokens:
        return 0.0
    pin, pout = get_prices(llm_provider, model)
    return (int(input_tokens) * pin + int(output_tokens) * pout) / 1_000_000.0
