"""Agent trajectory logger — JSONL format.

Two-layer persistence:
- `trajectory.jsonl` — compact rows (human-readable / backward-compatible),
  keeps `tool_result_preview[:500]`
- `trajectory_full.jsonl` — sidecar with full prompt_sha256 / raw_response /
  tool_result_full; prompt text is deduped by sha256 into `prompts/<sha256>.txt`
  to support offline replay.

path_idx semantics: omitted for single_shot/diverse; written as int for hybrid.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


class TrajectoryLogger:
    """JSONL logger that records each agent tool-call turn."""

    def __init__(
        self,
        output_path: Path,
        full_log_enabled: bool = True,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = output_path
        self._file = output_path.open("a", encoding="utf-8")
        self._cumulative_cost = 0.0
        self._full_log_enabled = full_log_enabled
        self._full_file = None
        self._prompts_dir: Path | None = None
        if full_log_enabled:
            full_path = output_path.parent / "trajectory_full.jsonl"
            self._full_file = full_path.open("a", encoding="utf-8")
            self._prompts_dir = output_path.parent / "prompts"
            self._prompts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cumulative_cost(self) -> float:
        return self._cumulative_cost

    @property
    def output_path(self) -> Path:
        return self._path

    @property
    def output_dir(self) -> Path:
        return self._path.parent

    def _write_prompt_hashed(self, prompt_text: str) -> str:
        """Write prompt to disk keyed by sha256; identical prompts are written only once."""
        assert self._prompts_dir is not None
        digest = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        path = self._prompts_dir / f"{digest}.txt"
        if not path.exists():
            path.write_text(prompt_text, encoding="utf-8")
        return digest

    def log_turn(
        self,
        query_idx: int,
        doc_id: str,
        turn_index: int,
        agent_reasoning: str,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result_preview: str,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        path_idx: int | None = None,
        tool_result_full: Any = None,
        prompt_text: str | None = None,
        raw_response: str | None = None,
    ) -> None:
        """Record a single tool-call turn.

        Compact row → trajectory.jsonl.
        Full payload → trajectory_full.jsonl (includes prompt_sha256, raw_response,
        tool_result_full). path_idx is omitted when None (single_shot/diverse).
        """
        self._cumulative_cost += cost_usd
        record: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "query_idx": query_idx,
            "doc_id": doc_id,
            "turn_index": turn_index,
            "agent_reasoning": agent_reasoning,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result_preview": tool_result_preview[:500],
            "cost_usd": round(cost_usd, 6),
            "cumulative_cost_usd": round(self._cumulative_cost, 6),
            "latency_ms": round(latency_ms, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        if path_idx is not None:
            record["path_idx"] = path_idx
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

        if self._full_log_enabled and self._full_file is not None:
            full_record: dict[str, Any] = {
                "timestamp": record["timestamp"],
                "query_idx": query_idx,
                "doc_id": doc_id,
                "turn_index": turn_index,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_result_full": tool_result_full,
                "raw_response": raw_response,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            if path_idx is not None:
                full_record["path_idx"] = path_idx
            if prompt_text is not None:
                full_record["prompt_sha256"] = self._write_prompt_hashed(prompt_text)
                full_record["prompt_char_count"] = len(prompt_text)
            self._full_file.write(json.dumps(full_record, ensure_ascii=False, default=str) + "\n")
            self._full_file.flush()

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Record a run summary (last line). Caller may append arbitrary fields."""
        summary["_type"] = "summary"
        summary["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._file.write(json.dumps(summary, ensure_ascii=False) + "\n")
        self._file.flush()
        if self._full_log_enabled and self._full_file is not None:
            self._full_file.write(json.dumps(summary, ensure_ascii=False) + "\n")
            self._full_file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
        if self._full_file is not None and not self._full_file.closed:
            self._full_file.close()
