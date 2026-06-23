"""Standalone Azure client for the Evaporate baseline (isolated venv).

Imported from INSIDE `.venv-evaporate` by `run_variant.py`, which can't import the
repo's `models.*`. Reads creds from `local/azure.json` (model-selectable: gpt54 or
gpt54mini, mirroring src/models/gpt54*.py) and records every completion in a
phase-tagged ({synthesis, extraction}) usage ledger for two-column cost reporting.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from openai import AzureOpenAI

# src/baseline/evaporate/runtime/llm_backend.py -> repo root is parents[4]
_ROOT = Path(__file__).resolve().parents[4]
_AZURE_JSON = _ROOT / "local" / "azure.json"

VALID_PHASES = ("synthesis", "extraction")


def _parse_key_file(path: str | Path) -> dict[str, str]:
    """Parse a `key: value` text file (mirrors azure_local.load_*_from_key_file)."""
    cfg: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            cfg[k.strip()] = v.strip()
    return cfg


def _load_credentials(model: str) -> tuple[str, str, str, str]:
    """Return (api_key, api_version, endpoint, deployment) for `model`.

    gpt54mini → azure.json `key_file_cheap`; gpt54 → azure.json `key_file`/inline
    (mirrors src/models/gpt54*.py). Match the model to the compared LSF variant.
    """
    cfg = json.loads(_AZURE_JSON.read_text(encoding="utf-8"))
    if model == "gpt54mini":
        cheap = cfg.get("key_file_cheap", "")
        if not cheap:
            raise RuntimeError("key_file_cheap not set in local/azure.json")
        kf = _parse_key_file(cheap)
        api_key = kf.get("api_key", kf.get("key", ""))
        api_version = kf.get("api_version", kf.get("AZURE_OPENAI_API_VERSION", ""))
        endpoint = kf.get("azure_endpoint", kf.get("endpoint", kf.get("AZURE_OPENAI_ENDPOINT", "")))
        deployment = (kf.get("deployment", kf.get("model_name", kf.get("model", "gpt-5.4-mini"))) or "gpt-5.4-mini").strip()
        src = cheap
    elif model == "gpt54":
        # Mirror src/models/gpt54.py (load_azure_credentials_from_local): follow a
        # `key_file` pointer if present (overlay onto inline), else use inline values.
        merged = dict(cfg)
        kf = cfg.get("key_file", "")
        if kf:
            merged.update(_parse_key_file(kf))
        api_key = merged.get("api_key", merged.get("key", ""))
        api_version = merged.get("api_version", merged.get("AZURE_OPENAI_API_VERSION", ""))
        endpoint = merged.get("azure_endpoint", merged.get("endpoint", merged.get("AZURE_OPENAI_ENDPOINT", "")))
        deployment = (merged.get("deployment", merged.get("model_name", merged.get("model", "gpt-5.4"))) or "gpt-5.4").strip()
        src = kf or str(_AZURE_JSON)
    else:
        raise ValueError(f"unknown model {model!r} (expected 'gpt54' or 'gpt54mini')")
    if not api_key or not endpoint:
        raise RuntimeError(f"incomplete {model} credentials from {src}")
    return api_key, api_version, endpoint, deployment


class LLMBackend:
    """Thin Azure gpt54/gpt54mini wrapper with a phase-tagged usage ledger."""

    def __init__(self, model: str = "gpt54") -> None:
        self.model = model
        api_key, api_version, endpoint, deployment = _load_credentials(model)
        self.deployment = deployment
        self._client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
            timeout=120.0,
            max_retries=3,
        )
        self._phase = "extraction"
        self._lock = threading.Lock()
        # One record per completion: {phase, model, prompt_tokens, completion_tokens, latency_s}
        self.ledger: list[dict[str, Any]] = []

    # -- phase control -------------------------------------------------------
    def set_phase(self, phase: str) -> None:
        if phase not in VALID_PHASES:
            raise ValueError(f"phase must be one of {VALID_PHASES}, got {phase!r}")
        self._phase = phase

    # -- the single call surface --------------------------------------------
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_completion_tokens: int = 5000,
        temperature: float = 0.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        phase: str | None = None,
        stop: list[str] | None = None,
    ) -> tuple[str, int]:
        """One chat completion → (text, total_tokens). Sends gpt54*-style params
        (temperature=0, top_p=1, no penalties); records a phase-tagged ledger entry."""
        ph = phase or self._phase
        if ph not in VALID_PHASES:
            raise ValueError(f"phase must be one of {VALID_PHASES}, got {ph!r}")
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {
            "model": self.deployment,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        # gpt-5.4(-mini) reasoning deployments reject `stop` (400), so we never send
        # it — emulate it by post-truncating the completion below.
        t0 = time.time()
        resp = self._client.chat.completions.create(**kwargs)
        dt = time.time() - t0
        text = resp.choices[0].message.content or ""
        for s in (stop or []):
            idx = text.find(s)
            if idx != -1:
                text = text[:idx]
        text = text.strip()
        u = getattr(resp, "usage", None)
        pt = int(getattr(u, "prompt_tokens", 0) or 0) if u else 0
        ct = int(getattr(u, "completion_tokens", 0) or 0) if u else 0
        tt = int(getattr(u, "total_tokens", pt + ct) or (pt + ct)) if u else (pt + ct)
        with self._lock:
            self.ledger.append({
                "phase": ph,
                "model": self.model,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": tt,
                "latency_s": round(dt, 3),
            })
        return text, tt

    # -- ledger reporting ----------------------------------------------------
    def ledger_summary(self) -> dict[str, Any]:
        agg = {p: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
               for p in VALID_PHASES}
        for r in self.ledger:
            b = agg[r["phase"]]
            b["calls"] += 1
            b["prompt_tokens"] += r["prompt_tokens"]
            b["completion_tokens"] += r["completion_tokens"]
            b["total_tokens"] += r["total_tokens"]
        return {
            "model": self.model,
            "by_phase": agg,
            "n_calls": len(self.ledger),
            "credential_source": str(_AZURE_JSON),
        }


if __name__ == "__main__":
    # Smoke check: prove the credential plumbing + one live call work.
    be = LLMBackend()
    print("deployment:", be.deployment)
    be.set_phase("extraction")
    txt, toks = be.complete("Reply with the single word OK.", max_completion_tokens=10)
    print("response:", txt, "| total_tokens:", toks)
    print("ledger:", json.dumps(be.ledger_summary(), indent=2))
