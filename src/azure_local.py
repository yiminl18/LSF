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


def install_usage_logging(client, model_tag: str) -> None:
    """Opt-in per-call token logging on an AzureOpenAI client.

    Active ONLY when env ``LSF_LLM_USAGE_LOG`` points to a file (e.g. set by the
    pipeline for the rule-gen subprocess). Each ``chat.completions.create`` then
    appends one JSON line ``{"model","prompt_tokens","completion_tokens"}``. This
    captures an agent's LLM *verification* calls during rule-gen — the QA and
    judge calls (via rule_apply_merge / eval_rule) on the sampled AND held-out
    validation docs — which are separate Azure calls, not part of the Codex
    agent's own token stream. No-op when the env var is unset, so normal pipeline
    stages (apply/eval) are completely unaffected.
    """
    import os
    log_path = os.environ.get("LSF_LLM_USAGE_LOG")
    if not log_path:
        return
    _orig = client.chat.completions.create

    def _logged(*args, **kwargs):
        resp = _orig(*args, **kwargs)
        try:
            u = getattr(resp, "usage", None)
            if u is not None:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "model": model_tag,
                        "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
                        "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
                    }) + "\n")
        except Exception:
            pass
        return resp

    try:
        client.chat.completions.create = _logged
    except Exception:
        pass


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
