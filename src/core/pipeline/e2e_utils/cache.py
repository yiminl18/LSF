"""
LLM call cache module.

SQLite-backed thread-safe LLM call cache.
Each thread uses an independent SQLite connection to avoid concurrent access issues.
Cache keys include the model identifier to distinguish per-model responses.
"""

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from core.llm.model import llm_call
from core.llm.tokens import estimate_tokens

# Unified LLM cache path. All pipelines and agent entry points share this DB so
# identical (prompt, provider, model, ...) keys hit the same row regardless of
# which caller issued the request. Override per-call via CachedLLMCaller(db_path=...).
DEFAULT_CACHE_DB_PATH: str = ".cache/llm_cache.db"

_CREATE_TABLE_SQL = """
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


def _schema_identity(response_schema: dict | None) -> str:
    if response_schema is None:
        return ""
    return json.dumps(
        response_schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _normalize_temperature(temperature: float | int | None) -> float:
    if temperature is None:
        return 0.0
    return float(temperature)


def _use_cache_for_temperature(temperature: float | int | None) -> bool:
    return _normalize_temperature(temperature) == 0.0


def _require_model(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("CachedLLMCaller.call requires an explicit model")
    resolved_model = model.strip()
    if resolved_model == "unspec" + "ified":
        raise ValueError("CachedLLMCaller.call model uses a reserved invalid name")
    return resolved_model


def _build_cache_key(
    *,
    prompt: str,
    llm_provider: str,
    model: str,
    max_tokens: int,
    temperature: float,
    response_schema: dict | None,
) -> str:
    return hashlib.sha256(
        f"{prompt}|{llm_provider}|{model}|{max_tokens}|{temperature}|{_schema_identity(response_schema)}".encode()
    ).hexdigest()


@dataclass
class CacheResult:
    """LLM call result, including cache-hit flag."""

    response: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cache_hit: bool
    cached_input_tokens: int = 0
    cost_usd: float = 0.0


class CachedLLMCaller:
    """Thread-safe SQLite-backed LLM caller with caching.

    Each thread holds an independent SQLite connection (via threading.local),
    allowing concurrent calls without "database is locked" errors.
    """

    def __init__(self, db_path: str = DEFAULT_CACHE_DB_PATH) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._local = threading.local()
        # Initialize table schema from the main thread.
        conn = self._get_conn()
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """Get the SQLite connection for the current thread (lazy initialization)."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.execute(_CREATE_TABLE_SQL)
        return self._local.conn

    def call(
        self,
        prompt: str,
        llm_provider: str = "azure",
        max_tokens: int = 800,
        *,
        model: str,
        response_schema: dict | None = None,
        temperature: float = 0,
    ) -> CacheResult:
        """
        Call the LLM, returning a cached result on cache hit.

        Thread-safe: each thread uses an independent SQLite connection.

        Args:
            prompt: prompt text
            llm_provider: LLM provider
            max_tokens: maximum output tokens
            model: LLM model identifier, required
            response_schema: optional structured-output schema
            temperature: sampling temperature; non-zero disables caching by default

        Returns:
            CacheResult (cache_hit=True when served from cache)

        Raises:
            Exceptions raised by llm_call() propagate directly; failed calls are not cached.
        """
        conn = self._get_conn()
        normalized_temperature = _normalize_temperature(temperature)
        resolved_model = _require_model(model)
        cache_key = _build_cache_key(
            prompt=prompt,
            llm_provider=llm_provider,
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=normalized_temperature,
            response_schema=response_schema,
        )

        use_cache = _use_cache_for_temperature(normalized_temperature)
        if use_cache:
            row = conn.execute(
                "SELECT response, input_tokens, output_tokens, latency_ms "
                "FROM llm_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()

            if row is not None:
                response, input_tokens, output_tokens, latency_ms = row
                return CacheResult(
                    response=response,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    cache_hit=True,
                )

        # Cache miss — call the LLM.
        t0 = time.perf_counter()
        response = llm_call(
            prompt,
            llm_provider=llm_provider,
            max_tokens=max_tokens,
            model=resolved_model,
            response_schema=response_schema,
            temperature=normalized_temperature,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        input_tokens = estimate_tokens(prompt)
        output_tokens = estimate_tokens(response)

        if use_cache:
            conn.execute(
                """
                INSERT OR IGNORE INTO llm_cache
                    (cache_key, prompt_text, response, input_tokens, output_tokens,
                     latency_ms, model, llm_provider, max_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    prompt,
                    response,
                    input_tokens,
                    output_tokens,
                    latency_ms,
                    resolved_model,
                    llm_provider,
                    max_tokens,
                ),
            )
            conn.commit()

        return CacheResult(
            response=response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cache_hit=False,
        )
