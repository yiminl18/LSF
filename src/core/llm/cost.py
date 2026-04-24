"""
LLM 调用费用计算

基于 token 数量和 provider 定价计算 LLM API 调用费用。
所有 provider 的定价集中在此管理。
"""

from core.config import (
    GPT_PRICE_PER_MILLION_INPUT,
    GPT_PRICE_PER_MILLION_OUTPUT,
    GPT_54_MINI_PRICE_PER_MILLION_INPUT,
    GPT_54_MINI_PRICE_PER_MILLION_OUTPUT,
    GPT_54_PRICE_PER_MILLION_INPUT,
    GPT_54_PRICE_PER_MILLION_OUTPUT,
)

# provider → (input_price, output_price) per million tokens
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

# mini 模型定价（跨 provider 统一）
MINI_PRICE_INPUT = 0.15
MINI_PRICE_OUTPUT = 0.60


def compute_cost(
    input_tokens: int,
    output_tokens: int,
    llm_provider: str,
    model: str,
) -> float:
    """计算 LLM 调用费用（美元）。

    参数:
        input_tokens: 输入 token 数量
        output_tokens: 输出 token 数量
        llm_provider: 提供商名称（azure/openrouter）
        model: 模型名称（用于 provider 内部的模型级定价）

    返回:
        费用（美元）

    异常:
        ValueError: 未知的 provider
    """
    if input_tokens == 0 and output_tokens == 0:
        return 0.0

    price_input, price_output = get_prices(llm_provider, model=model)
    return (input_tokens * price_input + output_tokens * price_output) / 1_000_000


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


def get_prices(llm_provider: str, model: str) -> tuple[float, float]:
    """获取指定 provider/model 的 (input_price, output_price) per million tokens。

    参数:
        llm_provider: 提供商名称
        model: 模型名称（含 "mini" 时使用 mini 定价）

    返回:
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
        raise ValueError(f"未知的 llm_provider: {llm_provider!r}")

    return PROVIDER_PRICES[llm_provider]
