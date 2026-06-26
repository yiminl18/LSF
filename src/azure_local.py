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


def load_openai_key_from_local(
    azure_json_path: str | Path,
    *,
    json_key: str = "openai_key_file",
) -> tuple[str, str | None, str | None]:
    """Load a *general* (non-Azure) OpenAI key referenced from ``local/azure.json``.

    Reads ``cfg[json_key]`` (default ``openai_key_file``) — a path to a text file
    holding either a raw key (one line, e.g. ``sk-...``) or ``key: value`` lines
    (``api_key:``, optional ``base_url:`` / ``organization:``).

    Returns ``(api_key, base_url, organization)``; the last two are ``None`` if
    unspecified. Use with ``openai.OpenAI`` (not ``AzureOpenAI``).
    """
    cfg: dict[str, str] = json.loads(Path(azure_json_path).read_text())
    key_file = cfg.get(json_key)
    if not key_file:
        raise RuntimeError(
            f"'{json_key}' not set in {azure_json_path}. Add it pointing at a file "
            "containing the OpenAI api_key (raw or 'api_key: sk-...')."
        )

    api_key = ""
    base_url: str | None = None
    organization: str | None = None
    for line in Path(key_file).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line and not line.startswith("sk-") and not line.startswith("http"):
            k, _, v = line.partition(":")
            k, v = k.strip().lower(), v.strip()
            if k in ("api_key", "key", "openai_api_key"):
                api_key = v
            elif k in ("base_url", "endpoint", "openai_base_url"):
                base_url = v
            elif k in ("organization", "org", "openai_organization"):
                organization = v
        else:
            api_key = line  # raw single-line key

    if not api_key:
        raise RuntimeError(f"No api_key found in {key_file}")
    return api_key, base_url, organization


# ---------------------------------------------------------------------------
# Provider switch — Azure vs general OpenAI, selected at runtime.
# ---------------------------------------------------------------------------

# Role -> (azure key-file json field, azure default deployment,
#          openai model json field, openai default model)
_ROLE_SPEC = {
    "chat_large": ("key_file",          "gpt-5.4",                "openai_model",           "gpt-5.4"),
    "chat_mini":  ("key_file_cheap",    "gpt-5.4-mini",           "openai_model_cheap",     "gpt-5.4-mini"),
    "embedding":  ("embedding_key_file", "text-embedding-3-small", "openai_embedding_model", "text-embedding-3-small"),
}


def get_provider(cfg: dict | None = None) -> str:
    """Active LLM provider: ``openai`` or ``azure`` (default).

    Resolution order (first wins): env ``LSF_LLM_PROVIDER`` -> ``cfg["provider"]``
    (the ``provider`` field in ``local/azure.json``) -> ``azure``. Env always
    overrides the file, so a one-off run can flip provider without editing config.
    Routes every model module — and thus every pipeline/baseline/strategy that
    imports them — through the chosen key.
    """
    import os
    env = os.environ.get("LSF_LLM_PROVIDER")
    if env:
        return env.strip().lower()
    if cfg and cfg.get("provider"):
        return str(cfg["provider"]).strip().lower()
    return "azure"


def build_model_client(
    azure_json_path: str | Path,
    role: str,
    *,
    timeout: float = 600.0,
    max_retries: int = 3,
):
    """Build the client + model name for a role under the active provider.

    ``role`` is one of ``chat_large`` / ``chat_mini`` / ``embedding``. Returns a
    dict ``{client, model, provider, api_key, endpoint, api_version}``. Both the
    Azure and OpenAI clients expose identical ``chat.completions.create`` /
    ``embeddings.create`` interfaces, so ``client`` is a drop-in either way; only
    the ``model`` string differs (Azure deployment name vs public OpenAI id).
    """
    if role not in _ROLE_SPEC:
        raise ValueError(f"unknown role {role!r}; expected one of {list(_ROLE_SPEC)}")
    azure_field, azure_default, openai_field, openai_default = _ROLE_SPEC[role]

    cfg: dict[str, str] = json.loads(Path(azure_json_path).read_text())
    provider = get_provider(cfg)

    if provider == "openai":
        from openai import OpenAI
        api_key, base_url, organization = load_openai_key_from_local(azure_json_path)
        model = (cfg.get(openai_field) or openai_default).strip()
        client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            organization=organization or None,
            timeout=timeout,
            max_retries=max_retries,
        )
        return {
            "client": client,
            "model": model,
            "provider": "openai",
            "api_key": api_key,
            "endpoint": base_url or "https://api.openai.com/v1",
            "api_version": "",
        }

    if provider not in ("azure", ""):
        raise ValueError(
            f"LSF_LLM_PROVIDER={provider!r} not supported; use 'openai' or 'azure'."
        )

    # Azure path — chat_large reads inline-or-key_file via the local loader;
    # the other roles read their dedicated key file directly.
    from openai import AzureOpenAI
    if role == "chat_large":
        api_key, api_version, endpoint, deployment = load_azure_credentials_from_local(azure_json_path)
    else:
        key_file = cfg.get(azure_field, "")
        if not key_file:
            raise RuntimeError(f"'{azure_field}' not set in {azure_json_path} (needed for role {role}).")
        api_key, api_version, endpoint, deployment = load_azure_credentials_from_key_file(key_file)
    model = (deployment or azure_default).strip()
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )
    return {
        "client": client,
        "model": model,
        "provider": "azure",
        "api_key": api_key,
        "endpoint": endpoint,
        "api_version": api_version,
    }
