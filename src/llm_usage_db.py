"""SQLite-backed LLM call recorder + cache (ported/adapted from LSF-dev
core/llm/pipeline cache).

One row per successful temperature-0 LLM call: prompt, response, input/output
tokens (API-reported when available, tiktoken fallback), latency, model, provider.
Serves cached responses on temperature-0 cache hits, so re-runs are near-free and
every paid call is durably recorded.

`wrap_openai_create(client, provider=..., model_default=...)` monkeypatches an
OpenAI/AzureOpenAI client's `chat.completions.create` to record+cache. Usable from
both the main env and the isolated `.venv-evaporate` (stdlib sqlite3; tiktoken
optional). DB is shared across processes (WAL + busy_timeout); each thread/process
opens its own connection.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_CREATE = """
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key    TEXT PRIMARY KEY,
    prompt_text  TEXT,
    response     TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    latency_ms   REAL,
    model        TEXT,
    llm_provider TEXT,
    max_tokens   INTEGER,
    timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""


def _estimate_tokens(text: str) -> int:
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text or ""))
    except Exception:
        return max(1, len((text or "")) // 4)  # rough char/4 fallback


def _cache_key(prompt: str, provider: str, model: str, max_tokens: int, temperature: float) -> str:
    return hashlib.sha256(
        f"{prompt}|{provider}|{model}|{max_tokens}|{temperature}".encode()
    ).hexdigest()


class LLMUsageDB:
    """Thread- and process-safe SQLite recorder/cache (one connection per thread)."""

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_path)
        self._local = threading.local()
        self._conn().execute(_CREATE)
        self._conn().commit()

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self._db_path, timeout=30.0)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA busy_timeout=30000")
            c.execute(_CREATE)
            self._local.conn = c
        return c

    def lookup(self, key: str):
        return self._conn().execute(
            "SELECT response, input_tokens, output_tokens, latency_ms "
            "FROM llm_cache WHERE cache_key = ?", (key,),
        ).fetchone()

    def record(self, key: str, prompt: str, response: str, itok: int, otok: int,
               latency_ms: float, model: str, provider: str, max_tokens: int) -> None:
        c = self._conn()
        c.execute(
            "INSERT OR IGNORE INTO llm_cache (cache_key, prompt_text, response, "
            "input_tokens, output_tokens, latency_ms, model, llm_provider, max_tokens) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (key, prompt, response, int(itok), int(otok), float(latency_ms),
             model, provider, int(max_tokens or 0)),
        )
        c.commit()


def _stub_response(content: str, itok: int, otok: int) -> Any:
    """Minimal stand-in matching the bits callers read on a cache hit."""
    usage = SimpleNamespace(prompt_tokens=int(itok), completion_tokens=int(otok),
                            total_tokens=int(itok) + int(otok),
                            completion_tokens_details=None)
    msg = SimpleNamespace(content=content, role="assistant")
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                           usage=usage, _cache_hit=True)


def wrap_openai_create(client, *, provider: str, model_default: str, db_path: str):
    """Monkeypatch client.chat.completions.create to record+cache (temp==0 only)."""
    db = LLMUsageDB(db_path)
    orig = client.chat.completions.create

    def wrapped(**kwargs):
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", model_default)
        temperature = float(kwargs.get("temperature", 0) or 0)
        max_tokens = int(kwargs.get("max_completion_tokens")
                         or kwargs.get("max_tokens") or 0)
        prompt = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        key = _cache_key(prompt, provider, model, max_tokens, temperature)
        cacheable = temperature == 0.0

        if cacheable:
            row = db.lookup(key)
            if row is not None:
                resp, itok, otok, _lat = row
                return _stub_response(resp, itok, otok)

        t0 = time.perf_counter()
        resp = orig(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        try:
            content = resp.choices[0].message.content or ""
            u = getattr(resp, "usage", None)
            itok = int(getattr(u, "prompt_tokens", 0) or 0) if u else 0
            otok = int(getattr(u, "completion_tokens", 0) or 0) if u else 0
            if not itok:
                itok = _estimate_tokens(prompt)
            if not otok:
                otok = _estimate_tokens(content)
            if cacheable:
                db.record(key, prompt, content, itok, otok, latency_ms,
                          model, provider, max_tokens)
        except Exception:
            pass  # never let recording break the real call
        return resp

    client.chat.completions.create = wrapped
    return db
