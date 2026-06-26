"""General (non-Azure) OpenAI client — keyed from ``local/azure.json::openai_key_file``.

Unlike ``gpt54.py`` / ``gpt54mini.py`` / ``embedding3small.py`` (which use
``AzureOpenAI`` + per-model Azure deployments), this module uses the plain
``openai.OpenAI`` client with a single general OpenAI API key, so any model the
account can access is reachable by its public model id (e.g. ``gpt-5.4``,
``gpt-5.4-mini``, ``text-embedding-3-small``).

Usage:
    from models.openai_general import chat, embed, MODEL_GPT54, MODEL_GPT54_MINI
    chat("Say hi", model=MODEL_GPT54)
    embed(["hello", "world"])                 # text-embedding-3-small
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

_SRC = Path(__file__).resolve().parents[1]
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from azure_local import load_openai_key_from_local  # noqa: E402

# Public model ids (override per call via the `model=` arg if your account
# exposes different names).
MODEL_GPT54 = "gpt-5.4"
MODEL_GPT54_MINI = "gpt-5.4-mini"
MODEL_EMBED = "text-embedding-3-small"

_AZURE_JSON = _ROOT / "local" / "azure.json"
api_key, _base_url, _org = load_openai_key_from_local(_AZURE_JSON)

client = OpenAI(
    api_key=api_key,
    base_url=_base_url or None,
    organization=_org or None,
    timeout=600.0,
    max_retries=3,
)


def chat(
    prompt: str,
    *,
    system: str | None = None,
    model: str = MODEL_GPT54,
    max_completion_tokens: int = 5000,
    temperature: float | None = None,
    **kwargs: Any,
) -> str:
    """Chat completion; optional ``system`` then ``prompt``. Returns message text.

    ``temperature`` is omitted unless set (some reasoning models reject non-default
    values). Extra kwargs pass straight to ``chat.completions.create``.
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    create_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }
    if temperature is not None:
        create_kwargs["temperature"] = temperature
    create_kwargs.update(kwargs)

    r = client.chat.completions.create(**create_kwargs)
    return (r.choices[0].message.content or "").strip()


def embed(
    texts: str | list[str],
    *,
    model: str = MODEL_EMBED,
    batch_size: int = 64,
) -> list[list[float]]:
    """Embed one or many strings; returns one vector per input (batched)."""
    if isinstance(texts, str):
        texts = [texts]
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        resp = client.embeddings.create(model=model, input=texts[i:i + batch_size])
        out.extend(d.embedding for d in resp.data)
    return out


def test(verbose: bool = True) -> dict[str, Any]:
    """Smoke-test gpt-5.4 chat, gpt-5.4-mini chat, and the embedding model.

    Returns a dict with per-model status; raises nothing — each model is tried
    independently so one failure doesn't mask the others.
    """
    results: dict[str, Any] = {}

    for tag, model in (("gpt5.4", MODEL_GPT54), ("gpt5.4-mini", MODEL_GPT54_MINI)):
        try:
            ans = chat(
                "Reply with exactly one word: pong.",
                system="You are a terse assistant.",
                model=model,
                max_completion_tokens=16,
            )
            results[tag] = {"ok": True, "model": model, "reply": ans}
            if verbose:
                print(f"[OK]   {tag:12s} ({model}) -> {ans!r}")
        except Exception as exc:  # noqa: BLE001
            results[tag] = {"ok": False, "model": model, "error": f"{type(exc).__name__}: {exc}"}
            if verbose:
                print(f"[FAIL] {tag:12s} ({model}) -> {type(exc).__name__}: {exc}")

    try:
        vecs = embed(["hello world", "the quick brown fox"], model=MODEL_EMBED)
        dim = len(vecs[0]) if vecs else 0
        results["embedding"] = {"ok": True, "model": MODEL_EMBED, "n": len(vecs), "dim": dim}
        if verbose:
            print(f"[OK]   {'embedding':12s} ({MODEL_EMBED}) -> {len(vecs)} vectors, dim={dim}")
    except Exception as exc:  # noqa: BLE001
        results["embedding"] = {"ok": False, "model": MODEL_EMBED, "error": f"{type(exc).__name__}: {exc}"}
        if verbose:
            print(f"[FAIL] {'embedding':12s} ({MODEL_EMBED}) -> {type(exc).__name__}: {exc}")

    return results


if __name__ == "__main__":
    print("azure.json:", _AZURE_JSON)
    print("base_url:  ", _base_url or "(default api.openai.com)")
    print("api_key:   ", (api_key[:8] + "…") if len(api_key) > 8 else "***")
    print("---")
    res = test()
    print("---")
    ok = sum(1 for v in res.values() if v.get("ok"))
    print(f"summary: {ok}/{len(res)} models OK")
    sys.exit(0 if ok == len(res) else 1)
