"""Azure OpenAI text-embedding-3-small client.

Reads `embedding_key_file` from `local/azure.json` and constructs an Azure
client pointed at the embedding deployment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from azure_local import load_azure_credentials_from_key_file  # noqa: E402
from openai import AzureOpenAI  # noqa: E402

_AZURE_JSON = _ROOT / "local" / "azure.json"
_cfg = json.loads(_AZURE_JSON.read_text())
_key_file = _cfg.get("embedding_key_file")
if not _key_file:
    raise RuntimeError(
        f"'embedding_key_file' not set in {_AZURE_JSON}. "
        "Add it pointing at the file with api_key/api_version/endpoint/deployment "
        "for the embedding model."
    )

api_key, AZURE_API_VERSION, AZURE_ENDPOINT, _deployment = load_azure_credentials_from_key_file(_key_file)
AZURE_DEPLOYMENT = (_deployment or "text-embedding-3-small").strip()
deployment = AZURE_DEPLOYMENT

client = AzureOpenAI(
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=api_key,
)


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
    print("azure.json:    ", _AZURE_JSON)
    print("key_file:      ", _key_file)
    print("endpoint:      ", AZURE_ENDPOINT)
    print("api_version:   ", AZURE_API_VERSION)
    print("deployment:    ", AZURE_DEPLOYMENT)
    v = embed("hello world")
    print(f"smoke test: 1 text → {len(v)} vector(s), dim={len(v[0])}")
