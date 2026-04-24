# -*- coding: utf-8 -*-
"""Azure GPT-5.4 wrapper."""

from __future__ import annotations

import os
import threading
from typing import Any

import tiktoken
from openai import AzureOpenAI

from core.llm.cost import get_prices

DEFAULT_MODEL = "gpt-5.4"
_ENV_PREFIX = "AZURE_54"

_cumulative_cost_usd = 0.0
_cost_lock = threading.Lock()


def estimate_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def run_azure_gpt54_family_request(
    *,
    prompt: str,
    max_tokens: int,
    temperature: float,
    model: str,
    response_schema: dict[str, Any] | None,
    env_prefix: str,
) -> tuple[str, int, int]:
    api_key = os.environ.get(f"{env_prefix}_API_KEY")
    azure_endpoint = os.environ.get(f"{env_prefix}_API_BASE")
    api_version = os.environ.get(f"{env_prefix}_API_VERSION")
    deployment = os.environ.get(f"{env_prefix}_DEPLOYMENT")

    missing = [
        name
        for name, value in {
            f"{env_prefix}_API_KEY": api_key,
            f"{env_prefix}_API_BASE": azure_endpoint,
            f"{env_prefix}_API_VERSION": api_version,
            f"{env_prefix}_DEPLOYMENT": deployment,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    request_kwargs: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "model": deployment,
    }
    if response_schema is not None:
        request_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": response_schema,
        }

    response = client.chat.completions.create(**request_kwargs)
    answer = _extract_response_text(response)
    return (
        answer,
        estimate_tokens(prompt, model=model),
        estimate_tokens(answer, model=model),
    )


def gpt_54_azure(
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


def _extract_response_text(response: Any) -> str:
    choice = response.choices[0]
    answer = _coerce_text_like(getattr(choice.message, "content", None))
    if answer is None:
        raise ValueError("Azure GPT-5.4 returned empty content")
    return answer


def _coerce_text_like(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None

    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            nested = _coerce_text_like(item)
            if nested is not None:
                texts.append(nested)
        if texts:
            return "\n".join(texts)

    if isinstance(value, dict):
        for key in ("text", "content", "output_text"):
            nested = _coerce_text_like(value.get(key))
            if nested is not None:
                return nested

    for attr in ("text", "content", "output_text"):
        if hasattr(value, attr):
            nested = _coerce_text_like(getattr(value, attr))
            if nested is not None:
                return nested

    return None
