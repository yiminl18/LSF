"""Load Azure OpenAI settings from ``local/azure.json`` and optional ``key_file``."""
from __future__ import annotations

import json
from pathlib import Path


def load_azure_credentials_from_local(
    azure_json_path: str | Path,
) -> tuple[str, str, str, str | None]:
    """Return (api_key, api_version, azure_endpoint, deployment)."""
    cfg: dict[str, str] = json.loads(Path(azure_json_path).read_text())

    key_file = cfg.get("key_file", "")
    if key_file:
        for line in Path(key_file).read_text().splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                cfg[k.strip()] = v.strip()

    api_key = cfg.get("api_key", cfg.get("key", ""))
    api_version = cfg.get("api_version", cfg.get("AZURE_OPENAI_API_VERSION", ""))
    endpoint = cfg.get("azure_endpoint", cfg.get("endpoint", cfg.get("AZURE_OPENAI_ENDPOINT", "")))
    deployment = cfg.get("deployment", cfg.get("model_name", cfg.get("model", None)))

    return api_key, api_version, endpoint, deployment


def load_azure_credentials_from_key_file(
    key_file_path: str | Path,
) -> tuple[str, str, str, str | None]:
    """Load credentials directly from a key file (api_key: ..., api_version: ..., etc.)."""
    cfg: dict[str, str] = {}
    for line in Path(key_file_path).read_text().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            cfg[k.strip()] = v.strip()

    api_key = cfg.get("api_key", cfg.get("key", ""))
    api_version = cfg.get("api_version", cfg.get("AZURE_OPENAI_API_VERSION", ""))
    endpoint = cfg.get("azure_endpoint", cfg.get("endpoint", cfg.get("AZURE_OPENAI_ENDPOINT", "")))
    deployment = cfg.get("deployment", cfg.get("model_name", cfg.get("model", None)))

    return api_key, api_version, endpoint, deployment
