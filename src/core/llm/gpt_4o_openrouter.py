# -*- coding: utf-8 -*-
"""
OpenRouter GPT-4o API Wrapper

Calls GPT-4o model via the OpenRouter API with cost tracking.

Main functions:
- gpt_4o_openrouter(): Call GPT-4o model and generate a response
- estimate_tokens(): Estimate the token count of a text
- reset_cost_counter(): Reset the cumulative cost counter
- get_cumulative_cost(): Get the cumulative cost

Dependencies:
- openai: OpenAI SDK (OpenRouter-compatible)
- tiktoken: OpenAI token counting library

Environment variables:
- OPENROUTER_API_KEY: OpenRouter API key
"""

import os
import threading
from openai import OpenAI
import tiktoken

from core.config import (
    OPENROUTER_GPT4O_PRICE_PER_MILLION_INPUT,
    OPENROUTER_GPT4O_PRICE_PER_MILLION_OUTPUT,
)

DEFAULT_MODEL = "openai/gpt-4o"

_cumulative_cost_usd = 0.0
_cost_lock = threading.Lock()


def estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Estimate the token count for a given text.

    Uses tiktoken to compute the number of tokens for the specified model.

    Args:
        text: The text to estimate
        model: Tokenizer model name (default: gpt-4o)

    Returns:
        Estimated token count
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def gpt_4o_openrouter(
    prompt: str,
    max_tokens: int = 800,
    temperature: float = 0,
    estimate_cost: bool = True,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Call the OpenRouter GPT-4o API and estimate cost.

    Sends a prompt to the GPT-4o model, retrieves the response, and tracks API call cost.

    Args:
        prompt: Prompt text to send to the model
        max_tokens: Max response tokens (default: 800)
        temperature: Response randomness 0-1 (default: 0)
        estimate_cost: Whether to estimate and print cost (default: True)
        model: Model name (default: openai/gpt-4o, also supports openai/gpt-4o-mini etc.)

    Returns:
        The model's response content
    """
    global _cumulative_cost_usd

    price_input = OPENROUTER_GPT4O_PRICE_PER_MILLION_INPUT
    price_output = OPENROUTER_GPT4O_PRICE_PER_MILLION_OUTPUT

    if "mini" in model.lower():
        price_input = 0.15
        price_output = 0.60

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing required environment variable: OPENROUTER_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    input_tokens = estimate_tokens(prompt, model="gpt-4o")
    input_cost_usd = input_tokens * (price_input / 1_000_000)

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
        )
        answer = response.choices[0].message.content
    except Exception:
        raise

    output_tokens = estimate_tokens(answer, model="gpt-4o")
    output_cost_usd = output_tokens * (price_output / 1_000_000)

    total_cost_usd = input_cost_usd + output_cost_usd
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
    """Reset the cumulative cost counter."""
    global _cumulative_cost_usd
    with _cost_lock:
        _cumulative_cost_usd = 0.0


def get_cumulative_cost() -> float:
    """Get the cumulative cost in USD."""
    with _cost_lock:
        return _cumulative_cost_usd
