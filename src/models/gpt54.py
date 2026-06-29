"""gpt-5.4 chat. Prefers the OpenAI *platform* key (``OPENAI_LSF_ONLY_API_KEY``);
falls back to Azure (``local/azure.json``) when that env var is unset."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import httpx
from openai import AzureOpenAI, OpenAI

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from azure_local import load_azure_credentials_from_local

_AZURE_JSON = _ROOT / "local" / "azure.json"

# Provider switch (priority: Pioneer > OpenAI platform > Azure). Pioneer
# (`PIONEER_API_KEY`, OpenAI-compatible base_url, gpt-5.4, 1200/min rate limit) is
# preferred for high-concurrency experiments; OpenAI platform (`OPENAI_LSF_ONLY_API_KEY`)
# next; Azure (`local/azure.json`, now 401-dead) last. `trust_env=False` on the Pioneer
# client bypasses any HTTP(S)_PROXY env (the proxy can't route to api.pioneer.ai).
_PIONEER_KEY = os.environ.get("PIONEER_API_KEY")
_OPENAI_KEY = os.environ.get("OPENAI_LSF_ONLY_API_KEY")
if _PIONEER_KEY:
    PROVIDER = "pioneer"
    api_key, AZURE_API_VERSION, AZURE_ENDPOINT = _PIONEER_KEY, None, None
    AZURE_DEPLOYMENT = "gpt-5.4"
    deployment = AZURE_DEPLOYMENT
    client = OpenAI(api_key=api_key, base_url="https://api.pioneer.ai/v1",
                    timeout=600.0, max_retries=3, http_client=httpx.Client(trust_env=False))
elif _OPENAI_KEY:
    PROVIDER = "openai"
    api_key, AZURE_API_VERSION, AZURE_ENDPOINT = _OPENAI_KEY, None, None
    AZURE_DEPLOYMENT = "gpt-5.4"
    deployment = AZURE_DEPLOYMENT
    client = OpenAI(api_key=api_key, timeout=600.0, max_retries=3)
else:
    PROVIDER = "azure"
    api_key, AZURE_API_VERSION, AZURE_ENDPOINT, _deployment = load_azure_credentials_from_local(
        _AZURE_JSON
    )
    AZURE_DEPLOYMENT = (_deployment or "gpt-5.4").strip()
    deployment = AZURE_DEPLOYMENT
    client = AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=api_key,
        timeout=600.0,     # large llm_coarse rule-gen prompts on big finance docs need >120s
        max_retries=3,     # retry transient failures / timeouts
    )

from azure_local import install_usage_logging as _install_usage_logging
_install_usage_logging(client, "gpt54")

# Per-call SQLite recorder + temperature-0 cache (shared .cache/llm_cache.db).
# Captures every call through this client — judge, chat_completions, pipeline
# rule-gen — with input/output tokens, latency, provider, model.
LLM_DB = None  # set below; exposes `.totals` (cumulative tokens, cache-aware)
try:
    from llm_usage_db import wrap_openai_create as _wrap_llm_db
    LLM_DB = _wrap_llm_db(client, provider=PROVIDER, model_default=AZURE_DEPLOYMENT,
                          db_path=str(_ROOT / ".cache" / "llm_cache.db"))
except Exception:
    pass  # recorder is best-effort; never block real calls


def chat_completions(
    prompt: str,
    *,
    system: str | None = None,
    max_completion_tokens: int = 5000,
    temperature: float = 0.0,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    model: str | None = None,
    **kwargs: Any,
) -> str:
    """Low-level Azure chat; optional ``system`` then ``prompt``. Extra kwargs go to ``create``."""
    m = model or AZURE_DEPLOYMENT
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    create_kwargs: dict[str, Any] = {
        "model": m,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }
    create_kwargs.update(kwargs)
    r = client.chat.completions.create(**create_kwargs)
    return (r.choices[0].message.content or "").strip()


def gpt_54(
    question: str,
    context: str,
    *,
    max_completion_tokens: int = 5000,
    **kwargs: Any,
) -> str:
    """Answer ``question`` using only ``context``; returns the assistant message text."""
    system = (
        "Answer using only the provided context. If it does not contain enough information, say so briefly."
    )
    user = f"Context:\n{context}\n\nQuestion:\n{question}"
    return chat_completions(
        user,
        system=system,
        max_completion_tokens=max_completion_tokens,
        **kwargs,
    )


if __name__ == "__main__":
    print("azure.json:", _AZURE_JSON)
    print("endpoint:", AZURE_ENDPOINT)
    print("api_version:", AZURE_API_VERSION)
    print("deployment:", AZURE_DEPLOYMENT)
    print("api_key:", (api_key[:8] + "…") if len(api_key) > 8 else "***")
    print("---")

    r = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I am going to Paris, what should I see?"},
        ],
        max_completion_tokens=16384,
    )
    print(r.choices[0].message.content)
