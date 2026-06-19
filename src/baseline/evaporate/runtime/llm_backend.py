"""Standalone Azure gpt54mini client for the Evaporate baseline (isolated venv).

This module is imported from *inside* `.venv-evaporate` by `run_variant.py`. That
venv cannot import the main repo's `models.*` (different, old dependency set), so
this file reads `local/azure.json` directly and replicates the gpt54mini call
semantics defined in `src/models/gpt54mini.py`:

  * credentials come from `key_file_cheap` in `local/azure.json`
    (a text file with `api_key`, `api_version`, `azure_endpoint`, `deployment`);
  * `max_completion_tokens=5000`, `temperature=0.0` defaults;
  * the deployment defaults to `gpt-5.4-mini`.

Every completion is recorded in a *phase-tagged* usage ledger so the orchestrator
can report Evaporate's synthesis cost (function generation) and apply/extraction
cost separately (plan Gap C, two columns). The phase is one of
{"synthesis", "extraction"}; set it with `set_phase(...)` or per call.

`local/azure.json` is the ONLY credential source (acceptance criterion #6).
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
    """Return (api_key, api_version, endpoint, deployment) for the requested model.

    `gpt54mini` follows `local/azure.json`'s `key_file_cheap` pointer (mirrors
    `src/models/gpt54mini.py`); `gpt54` reads the inline credentials + main
    `deployment` from `local/azure.json` (mirrors `src/models/gpt54.py`). The model
    must match the LSF variant the baseline is compared against.
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
        """Run one chat completion; return (text, total_tokens).

        Call params mirror src/models/gpt54mini.py's `chat_completions` EXACTLY
        (the semantic reference for the fair-comparison backend): it sends
        `max_completion_tokens`, `temperature=0.0`, `top_p=1.0`,
        `frequency_penalty=0.0`, `presence_penalty=0.0`. Records a phase-tagged
        ledger entry. `phase` defaults to the current phase set via `set_phase`.
        """
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
        # NOTE: gpt-5.4(-mini) reasoning deployments REJECT the `stop` parameter
        # (400 "Unsupported parameter: 'stop' is not supported with this model").
        # gpt54mini.py never sends it either. So we never pass `stop` to the API;
        # instead we emulate it by post-truncating the completion at the first
        # stop sequence below — same effect, no unsupported param.
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
