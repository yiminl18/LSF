# -*- coding: utf-8 -*-
"""Azure GPT-5.4-mini wrapper."""

from __future__ import annotations

import threading
from typing import Any

from core.llm.cost import get_prices
from core.llm.gpt_54_azure import (
    estimate_tokens as _estimate_tokens,
    run_azure_gpt54_family_request,
)

DEFAULT_MODEL = "gpt-5.4-mini"
_ENV_PREFIX = "AZURE_54MINI"

_cumulative_cost_usd = 0.0
_cost_lock = threading.Lock()


def estimate_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    return _estimate_tokens(text, model=model)


def gpt_54mini_azure(
    prompt: str,
    max_tokens: int = 800,
    temperature: float = 0,
    estimate_cost: bool = True,
    model: str = DEFAULT_MODEL,
    response_schema: dict[str, Any] | None = None,
) -> str:
    global _cumulative_cost_usd

    price_input, price_output = get_prices("azure", model=model)
    answer, input_tokens, output_tokens = run_azure_gpt54_family_request(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        model=model,
        response_schema=response_schema,
        env_prefix=_ENV_PREFIX,
    )

    total_cost_usd = input_tokens * (price_input / 1_000_000)
    total_cost_usd += output_tokens * (price_output / 1_000_000)
    with _cost_lock:
        _cumulative_cost_usd += total_cost_usd
        current_cumulative = _cumulative_cost_usd

    if estimate_cost:
        print(
            f"[COST] Input tokens: {input_tokens} | Output tokens: {output_tokens} | Cost: ${total_cost_usd:.6f}"
        )
        print(f"[COST_SUM] Cumulative cost: ${current_cumulative:.6f}")

    return answer


def reset_cost_counter() -> None:
    global _cumulative_cost_usd
    with _cost_lock:
        _cumulative_cost_usd = 0.0


def get_cumulative_cost() -> float:
    with _cost_lock:
        return _cumulative_cost_usd
