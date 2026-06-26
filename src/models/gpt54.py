"""Chat — gpt-5.4. Provider (Azure vs general OpenAI) chosen by ``LSF_LLM_PROVIDER``.

Azure (default): ``local/azure.json`` inline creds or ``key_file``.
``LSF_LLM_PROVIDER=openai`` -> general key in ``local/azure.json::openai_key_file``.
The public interface (``client``, ``chat_completions``, ``gpt_54``, ``AZURE_DEPLOYMENT``)
is unchanged, so callers/pipelines need no edits.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from azure_local import build_model_client, install_usage_logging as _install_usage_logging

_AZURE_JSON = _ROOT / "local" / "azure.json"

# large llm_coarse rule-gen prompts on big finance docs need >120s
_m = build_model_client(_AZURE_JSON, "chat_large", timeout=600.0, max_retries=3)
client = _m["client"]
PROVIDER = _m["provider"]
AZURE_DEPLOYMENT = _m["model"]
deployment = AZURE_DEPLOYMENT
api_key = _m["api_key"]
AZURE_ENDPOINT = _m["endpoint"]
AZURE_API_VERSION = _m["api_version"]

_install_usage_logging(client, "gpt54")


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
    print("provider:", PROVIDER)
    print("azure.json:", _AZURE_JSON)
    print("endpoint:", AZURE_ENDPOINT)
    print("api_version:", AZURE_API_VERSION or "(n/a)")
    print("model/deployment:", AZURE_DEPLOYMENT)
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
