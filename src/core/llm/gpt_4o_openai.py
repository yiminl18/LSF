# -*- coding: utf-8 -*-
"""
OpenAI GPT-4o API Wrapper (direct)

Environment variables:
- OPENAI_API_KEY: OpenAI API key
"""

import os
import threading
import time

import tiktoken
from openai import OpenAI, RateLimitError

from core.config import GPT_PRICE_PER_MILLION_INPUT, GPT_PRICE_PER_MILLION_OUTPUT

DEFAULT_MODEL = "gpt-4o"

_cumulative_cost_usd = 0.0
_cost_lock = threading.Lock()


def estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def gpt_4o_openai(
    prompt: str,
    max_tokens: int = 800,
    temperature: float = 0,
    estimate_cost: bool = True,
    model: str = DEFAULT_MODEL,
) -> str:
    global _cumulative_cost_usd

    price_input = GPT_PRICE_PER_MILLION_INPUT
    price_output = GPT_PRICE_PER_MILLION_OUTPUT

    if "mini" in model.lower():
        price_input = 0.15
        price_output = 0.60

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing required environment variable: OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    input_tokens = estimate_tokens(prompt, model="gpt-4o")
    input_cost_usd = input_tokens * (price_input / 1_000_000)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                model=model,
            )
            break
        except RateLimitError:
            if attempt < max_retries - 1:
                wait = 2**attempt * 5  # 5s, 10s, 20s
                print(
                    f"[RATE_LIMIT] 429 retry {attempt + 1}/{max_retries}, waiting {wait}s..."
                )
                time.sleep(wait)
            else:
                raise
    answer = response.choices[0].message.content

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
    global _cumulative_cost_usd
    with _cost_lock:
        _cumulative_cost_usd = 0.0


def get_cumulative_cost() -> float:
    with _cost_lock:
        return _cumulative_cost_usd
