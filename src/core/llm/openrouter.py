# -*- coding: utf-8 -*-
"""OpenRouter LLM client.

Exposes openrouter_chat() for LLM calls with cost tracking, plus
reset_cost_counter() / get_cumulative_cost() for per-run accounting. Uses
OPENROUTER_API_KEY. The model must be specified by the caller.
"""

import os
import json
import threading
from typing import Any

from openai import OpenAI

from core.llm.cost import get_prices
from core.llm.tokens import estimate_tokens

_MODELS_WITH_RESPONSE_FORMAT = {
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "z-ai/glm-5.1",
}

_cumulative_cost_usd = 0.0
_cost_lock = threading.Lock()


def _require_model(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("OpenRouter model must be specified explicitly")
    normalized_model = model.strip()
    if normalized_model == "unspec" + "ified":
        raise ValueError("OpenRouter model uses a reserved invalid name")
    return normalized_model


def openrouter_chat(
    prompt: str,
    max_tokens: int = 800,
    temperature: float = 0,
    estimate_cost: bool = True,
    *,
    model: str,
    response_schema: dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
) -> str:
    """
    调用 OpenRouter Chat Completions API 并估算成本。

    参数:
        prompt: 发送给模型的提示文本
        max_tokens: 响应的最大 token 数（默认: 800）
        temperature: 响应的随机性，0-1 之间（默认: 0）
        estimate_cost: 是否估算并打印成本（默认: True）
        model: 模型名称，必须显式指定
        response_schema: 可选 structured-output schema
        response_format: 可选 OpenRouter response_format

    返回:
        模型的响应内容
    """
    global _cumulative_cost_usd
    resolved_model = _require_model(model)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing required environment variable: OPENROUTER_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    request_kwargs: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "model": resolved_model,
    }
    structured_response_format = _build_response_format(
        model=resolved_model,
        response_schema=response_schema,
        response_format=response_format,
    )
    if structured_response_format is not None:
        request_kwargs["response_format"] = structured_response_format

    price_input, price_output = get_prices("openrouter", model=resolved_model)
    input_tokens = estimate_tokens(prompt)
    input_cost_usd = input_tokens * (price_input / 1_000_000)

    # 错误处理统一在 model.py:llm_call 层
    response = client.chat.completions.create(**request_kwargs)
    answer = _extract_response_text(
        response=response,
        model=resolved_model,
        response_schema=response_schema,
    )

    output_tokens = estimate_tokens(answer)
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


def _extract_response_text(
    response: Any,
    model: str,
    response_schema: dict[str, Any] | None,
) -> str:
    choice = response.choices[0]
    message = choice.message

    content = _coerce_text_like(getattr(message, "content", None))
    if content is not None:
        return content

    if not model.startswith("z-ai/glm-"):
        raise ValueError(_build_empty_content_error(choice, message))

    fallback_fields: list[tuple[str, Any]] = [
        ("reasoning_content", getattr(message, "reasoning_content", None)),
        ("reasoning", getattr(message, "reasoning", None)),
    ]

    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict):
        fallback_fields.extend(
            [
                ("model_extra.reasoning_content", model_extra.get("reasoning_content")),
                ("model_extra.reasoning", model_extra.get("reasoning")),
                ("model_extra.output_text", model_extra.get("output_text")),
            ]
        )

    for field_name, raw_value in fallback_fields:
        fallback_text = _coerce_text_like(raw_value)
        if fallback_text is None:
            continue
        if response_schema is not None:
            try:
                json.loads(fallback_text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"OpenRouter returned non-JSON fallback content via {field_name}"
                ) from exc
        return fallback_text

    raise ValueError(_build_empty_content_error(choice, message))


def _coerce_text_like(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None

    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    texts.append(stripped)
                continue
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            for key in ("text", "content"):
                text_value = item.get(key)
                if isinstance(text_value, str) and text_value.strip():
                    if item_type in {None, "text", "output_text"}:
                        texts.append(text_value.strip())
                        break
        if texts:
            return "\n".join(texts)

    if isinstance(value, dict):
        for key in ("text", "content", "output_text"):
            nested = _coerce_text_like(value.get(key))
            if nested is not None:
                return nested

    return None


def _build_empty_content_error(choice: Any, message: Any) -> str:
    finish_reason = getattr(choice, "finish_reason", None)
    available_fields = []
    for field_name in (
        "content",
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "tool_calls",
    ):
        if getattr(message, field_name, None) is not None:
            available_fields.append(field_name)
    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict):
        for field_name in (
            "reasoning",
            "reasoning_content",
            "output_text",
            "tool_calls",
        ):
            if model_extra.get(field_name) is not None:
                available_fields.append(f"model_extra.{field_name}")
    suffix = f" (finish_reason={finish_reason!r}, available_fields={available_fields})"
    return f"OpenRouter returned empty content{suffix}"


def reset_cost_counter() -> None:
    """重置累计成本计数器。"""
    global _cumulative_cost_usd
    with _cost_lock:
        _cumulative_cost_usd = 0.0


def get_cumulative_cost() -> float:
    """获取累计成本（美元）。"""
    with _cost_lock:
        return _cumulative_cost_usd


def _build_response_format(
    model: str,
    response_schema: dict[str, Any] | None,
    response_format: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if response_schema is not None and response_format is not None:
        raise ValueError("response_schema and response_format are mutually exclusive")

    resolved_response_format = response_format
    if response_schema is not None:
        resolved_response_format = {
            "type": "json_schema",
            "json_schema": response_schema,
        }

    if resolved_response_format is None:
        return None

    if model not in _MODELS_WITH_RESPONSE_FORMAT:
        raise ValueError(
            f"response_format is not supported for OpenRouter model={model!r}"
        )

    if not isinstance(resolved_response_format, dict):
        raise ValueError("response_format must be a dict")

    response_format_type = resolved_response_format.get("type")
    if response_format_type == "json_object":
        return resolved_response_format

    if response_format_type != "json_schema":
        raise ValueError(
            f"Unsupported OpenRouter response_format type={response_format_type!r} for model={model!r}"
        )

    json_schema = resolved_response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        raise ValueError(
            "OpenRouter json_schema response_format requires a dict json_schema"
        )
    if not json_schema.get("name"):
        raise ValueError("OpenRouter json_schema response_format requires schema name")
    if not isinstance(json_schema.get("schema"), dict):
        raise ValueError(
            "OpenRouter json_schema response_format requires dict schema body"
        )

    return resolved_response_format
