"""gpt-5.4-mini chat. Prefers the OpenAI *platform* key (``OPENAI_LSF_ONLY_API_KEY``)
or Pioneer (``PIONEER_API_KEY``); falls back to Azure (``key_file_cheap`` in
``local/azure.json``) when neither env var is set. Mirrors models/gpt54.py."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import AzureOpenAI, OpenAI

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from azure_local import load_azure_credentials_from_key_file

_AZURE_JSON = _ROOT / "local" / "azure.json"

# Provider switch (priority: Pioneer > OpenAI platform > Azure). gpt54mini ->
# platform model "gpt-5.4-mini". trust_env=False on Pioneer bypasses HTTP(S)_PROXY.
_PIONEER_KEY = os.environ.get("PIONEER_API_KEY")
_OPENAI_KEY = os.environ.get("OPENAI_LSF_ONLY_API_KEY")
if _PIONEER_KEY:
    import httpx
    PROVIDER = "pioneer"
    api_key, AZURE_API_VERSION, AZURE_ENDPOINT = _PIONEER_KEY, None, None
    AZURE_DEPLOYMENT = "gpt-5.4-mini"
    deployment = AZURE_DEPLOYMENT
    client = OpenAI(api_key=api_key, base_url="https://api.pioneer.ai/v1",
                    timeout=120.0, max_retries=3, http_client=httpx.Client(trust_env=False))
elif _OPENAI_KEY:
    PROVIDER = "openai"
    api_key, AZURE_API_VERSION, AZURE_ENDPOINT = _OPENAI_KEY, None, None
    AZURE_DEPLOYMENT = "gpt-5.4-mini"
    deployment = AZURE_DEPLOYMENT
    client = OpenAI(api_key=api_key, timeout=120.0, max_retries=3)
else:
    PROVIDER = "azure"
    _cfg = json.loads(_AZURE_JSON.read_text())
    _cheap_key_file = _cfg.get("key_file_cheap", "")
    if not _cheap_key_file:
        raise RuntimeError("key_file_cheap not set in local/azure.json")
    api_key, AZURE_API_VERSION, AZURE_ENDPOINT, _deployment = load_azure_credentials_from_key_file(
        _cheap_key_file
    )
    AZURE_DEPLOYMENT = (_deployment or "gpt-5.4-mini").strip()
    deployment = AZURE_DEPLOYMENT
    client = AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=api_key,
        timeout=120.0,
        max_retries=3,
    )

from azure_local import install_usage_logging as _install_usage_logging
_install_usage_logging(client, "gpt54mini")

# Per-call SQLite recorder + temperature-0 cache (shared .cache/llm_cache.db).
LLM_DB = None  # exposes `.totals` (cumulative tokens, cache-aware)
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


def gpt_54_mini(
    question: str,
    context: str,
    *,
    max_completion_tokens: int = 5000,
    **kwargs: Any,
) -> str:
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
            {"role": "user", "content": "What is 2+2?"},
        ],
        max_completion_tokens=50,
    )
    print(r.choices[0].message.content)
