"""LLM client helpers for the DeepRead baseline."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AzureOpenAI, OpenAI

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))

from azure_local import (
    load_azure_credentials_from_key_file,
    load_azure_credentials_from_local,
)
from baseline.cost import compute_cost

_AZURE_JSON = _ROOT / "local" / "azure.json"

DEFAULT_PROVIDER = "azure"
DEFAULT_MODEL = "gpt-5.4-mini"

_MODEL_ALIASES: dict[str, str] = {
    "gpt54": "gpt-5.4",
    "gpt54mini": "gpt-5.4-mini",
    "gpt-5.4": "gpt-5.4",
    "gpt-5.4-mini": "gpt-5.4-mini",
}


@dataclass(slots=True)
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    cost_usd: float
    model: str
    provider: str


def resolve_model(model: str | None) -> str:
    if not model:
        return DEFAULT_MODEL
    return _MODEL_ALIASES.get(model, model)


def _resolve_azure_credentials(model: str) -> tuple[str, str, str, str]:
    if "mini" in model.lower():
        import json

        cfg = json.loads(_AZURE_JSON.read_text())
        cheap_key_file = cfg.get("key_file_cheap", "")
        if not cheap_key_file:
            raise RuntimeError(
                f"key_file_cheap not set in {_AZURE_JSON}; cannot resolve {model} credentials"
            )
        api_key, api_version, endpoint, deployment = load_azure_credentials_from_key_file(
            cheap_key_file
        )
    else:
        api_key, api_version, endpoint, deployment = load_azure_credentials_from_local(
            _AZURE_JSON
        )

    if not api_key or not api_version or not endpoint:
        raise RuntimeError(
            f"Azure credentials incomplete for model={model!r}: "
            "need api_key, api_version, and azure_endpoint"
        )
    if not deployment or not deployment.strip():
        raise RuntimeError(
            f"Azure deployment name missing for model={model!r}; add `deployment:` "
            "to the corresponding local Azure key file."
        )
    return api_key, api_version, endpoint, deployment.strip()


def _usage_int(usage: Any, name: str) -> int:
    return int(getattr(usage, name, 0) or 0) if usage is not None else 0


def _finish_result(
    *,
    response: Any,
    provider: str,
    model: str,
    t0: float,
) -> LLMResult:
    text = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    input_tokens = _usage_int(usage, "prompt_tokens")
    output_tokens = _usage_int(usage, "completion_tokens")
    try:
        cost_usd = compute_cost(input_tokens, output_tokens, provider, model=model)
    except Exception:
        cost_usd = 0.0
    return LLMResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=round(time.perf_counter() - t0, 3),
        cost_usd=cost_usd,
        model=model,
        provider=provider,
    )


def _chat_create(
    messages: list[dict[str, Any]],
    *,
    provider: str,
    model: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> LLMResult:
    provider = (provider or DEFAULT_PROVIDER).strip().lower()
    model = resolve_model(model)
    t0 = time.perf_counter()

    if provider == "azure":
        api_key, api_version, endpoint, deployment = _resolve_azure_credentials(model)
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )
        return _finish_result(response=response, provider=provider, model=model, t0=t0)

    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for provider=openrouter")
        openrouter_model = model.removeprefix("openrouter:")
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        response = client.chat.completions.create(
            model=openrouter_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return _finish_result(
            response=response,
            provider=provider,
            model=openrouter_model,
            t0=t0,
        )

    raise ValueError(f"Unsupported DeepRead LLM provider: {provider!r}")


def chat_text(
    prompt: str,
    *,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 800,
    temperature: float = 0.0,
) -> LLMResult:
    return _chat_create(
        [{"role": "user", "content": prompt}],
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def chat_vision(
    prompt_text: str,
    image_b64: str,
    *,
    mime_type: str = "image/jpeg",
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.0,
) -> LLMResult:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_b64}",
                    },
                },
            ],
        }
    ]
    return _chat_create(
        messages,
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
