"""text-embedding-3-small client. Provider chosen by ``LSF_LLM_PROVIDER``.

Azure (default): ``embedding_key_file`` in ``local/azure.json``.
``LSF_LLM_PROVIDER=openai`` -> general key in ``local/azure.json::openai_key_file``.
Public interface (``client``, ``embed``, ``AZURE_DEPLOYMENT``) unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from azure_local import build_model_client  # noqa: E402

_AZURE_JSON = _ROOT / "local" / "azure.json"

_m = build_model_client(_AZURE_JSON, "embedding")
client = _m["client"]
PROVIDER = _m["provider"]
AZURE_DEPLOYMENT = _m["model"]
deployment = AZURE_DEPLOYMENT
api_key = _m["api_key"]
AZURE_ENDPOINT = _m["endpoint"]
AZURE_API_VERSION = _m["api_version"]


def embed(texts, model: str | None = None, batch_size: int = 64) -> list[list[float]]:
    """Embed one or many strings; returns list of vectors (one per input).

    Batches inputs in chunks of `batch_size` to stay within Azure request limits.
    """
    if isinstance(texts, str):
        texts = [texts]
    m = model or AZURE_DEPLOYMENT
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=m, input=batch)
        out.extend(d.embedding for d in resp.data)
    return out


if __name__ == "__main__":
    print("provider:      ", PROVIDER)
    print("azure.json:    ", _AZURE_JSON)
    print("endpoint:      ", AZURE_ENDPOINT)
    print("api_version:   ", AZURE_API_VERSION or "(n/a)")
    print("model/deploy:  ", AZURE_DEPLOYMENT)
    v = embed("hello world")
    print(f"smoke test: 1 text → {len(v)} vector(s), dim={len(v[0])}")
