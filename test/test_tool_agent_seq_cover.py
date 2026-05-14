from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agent.tool_agent.core import AgentConfig
from agent.tool_agent.document import DocumentContext
from agent.tool_agent.logger import TrajectoryLogger
from agent.tool_agent.seq_cover_core import run_seq_cover_agent_on_query
from core.pipeline.e2e_utils.cache import CacheResult


def _doc(doc_id: str, text: str, ground_truth: str) -> DocumentContext:
    entries = [
        {
            "label": "section_header",
            "text": "Cover",
            "page_no": 1,
            "structure": {"level": "H1", "parent_id": None},
        },
        {
            "label": "text",
            "text": text,
            "page_no": 1,
            "structure": {"level": "Body", "parent_id": 0},
        },
    ]
    return DocumentContext(
        doc_id=doc_id,
        query_idx=3,
        normalized_text=f"[Section] Cover\n{text}",
        ground_truth=ground_truth,
        entries=entries,
        section_index={0: entries[0]},
    )


def _rule_payload(anchor: str) -> dict[str, Any]:
    return {
        "rule_text": f"match {anchor}",
        "evidence_basis": "synthetic fixture",
        "retrieval_spec": {
            "mode": "regex",
            "anchor": anchor,
            "anchor_b": None,
            "page_idx": None,
            "max_chars": 64,
            "boundary_context_chars": 0,
        },
    }


def _generate_action(anchor: str) -> str:
    return json.dumps(
        {
            "action": "generate",
            "reasoning": "synthetic rule",
            "tool": None,
            "args": None,
            "rule": json.dumps(_rule_payload(anchor), ensure_ascii=False),
        },
        ensure_ascii=False,
    )


class _FakeCaller:
    def __init__(self, anchors: list[str]) -> None:
        self._anchors = list(anchors)
        self.prompts: list[str] = []

    def call(
        self,
        prompt: str,
        llm_provider: str = "azure",
        max_tokens: int = 800,
        *,
        model: str,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0,
    ) -> CacheResult:
        self.prompts.append(prompt)
        if response_schema is not None:
            anchor = self._anchors.pop(0) if self._anchors else "never"
            response = _generate_action(anchor)
        elif 'Return ONLY "True" or "False"' in prompt:
            reference = re.search(r"Reference answer: (.*)", prompt)
            generated = re.search(r"Generated answer: (.*)", prompt)
            response = (
                "True"
                if reference
                and generated
                and reference.group(1).strip() in generated.group(1).strip()
                else "False"
            )
        else:
            context = prompt.split("Context:\n---", 1)[1].split("---", 1)[0]
            response = next(
                (token for token in ("alpha", "beta", "gamma") if token in context),
                "Information not found.",
            )
        return CacheResult(
            response=response,
            input_tokens=10,
            output_tokens=5,
            latency_ms=1.0,
            cache_hit=False,
        )


def _config(**extra: Any) -> AgentConfig:
    cfg = AgentConfig(max_turns_per_query=1, budget_usd=10.0, agent_max_tokens=1000)
    for key, value in extra.items():
        setattr(cfg, key, value)
    return cfg


def test_seq_cover_terminates_on_full_coverage(tmp_path: Path) -> None:
    docs = [
        _doc("doc_a", "alpha", "alpha"),
        _doc("doc_b", "beta", "beta"),
        _doc("doc_c", "gamma", "gamma"),
    ]
    caller = _FakeCaller(["alpha", "beta", "gamma"])
    logger = TrajectoryLogger(tmp_path / "trajectory.jsonl")

    result = run_seq_cover_agent_on_query(
        query_text="find token",
        query_idx=3,
        doc_contexts=docs,
        cached_caller=caller,
        agent_config=_config(seq_cover_max_iterations=5),
        logger=logger,
        max_turns=1,
    )
    logger.close()

    assert [rule.retrieval_spec.anchor for rule in result.rules] == ["alpha", "beta", "gamma"]
    assert result.termination_reason == "full_coverage"
    trajectory = json.loads((tmp_path / "seq_cover_trajectory.json").read_text())
    assert trajectory["remaining_doc_ids"] == []


def test_seq_cover_respects_max_rules_cap(tmp_path: Path) -> None:
    docs = [_doc("doc_a", "alpha", "alpha"), _doc("doc_b", "beta", "beta")]
    caller = _FakeCaller(["alpha", "beta"])
    logger = TrajectoryLogger(tmp_path / "trajectory.jsonl")

    result = run_seq_cover_agent_on_query(
        query_text="find token",
        query_idx=3,
        doc_contexts=docs,
        cached_caller=caller,
        agent_config=_config(seq_cover_max_rules=1, seq_cover_max_iterations=5),
        logger=logger,
        max_turns=1,
    )
    logger.close()

    assert len(result.rules) == 1
    assert result.termination_reason == "max_rules"


def test_seq_cover_min_coverage_filter(tmp_path: Path) -> None:
    docs = [_doc("doc_a", "alpha", "alpha"), _doc("doc_b", "beta", "beta")]
    caller = _FakeCaller(["never"])
    logger = TrajectoryLogger(tmp_path / "trajectory.jsonl")

    result = run_seq_cover_agent_on_query(
        query_text="find token",
        query_idx=3,
        doc_contexts=docs,
        cached_caller=caller,
        agent_config=_config(seq_cover_max_iterations=3),
        logger=logger,
        max_turns=1,
    )
    logger.close()

    assert result.rules == []
    trajectory = json.loads((tmp_path / "seq_cover_trajectory.json").read_text())
    assert trajectory["iterations"][0]["accepted"] is False


def test_seq_cover_min_coverage_threshold_stops_immediately(tmp_path: Path) -> None:
    docs = [_doc("doc_a", "alpha", "alpha"), _doc("doc_b", "beta", "beta")]
    caller = _FakeCaller(["alpha", "beta"])
    logger = TrajectoryLogger(tmp_path / "trajectory.jsonl")

    result = run_seq_cover_agent_on_query(
        query_text="find token",
        query_idx=3,
        doc_contexts=docs,
        cached_caller=caller,
        agent_config=_config(
            seq_cover_max_iterations=3,
            seq_cover_min_coverage_threshold=2,
            seq_cover_min_marginal_coverage=1,
        ),
        logger=logger,
        max_turns=1,
    )
    logger.close()

    assert result.rules == []
    assert result.termination_reason == "min_coverage_threshold"
    trajectory = json.loads((tmp_path / "seq_cover_trajectory.json").read_text())
    assert len(trajectory["iterations"]) == 1


def test_seq_cover_signature_gate_rejects_duplicate(tmp_path: Path) -> None:
    docs = [_doc("doc_a", "alpha", "alpha"), _doc("doc_b", "beta", "beta")]
    caller = _FakeCaller(["alpha", "alpha", "beta"])
    logger = TrajectoryLogger(tmp_path / "trajectory.jsonl")

    result = run_seq_cover_agent_on_query(
        query_text="find token",
        query_idx=3,
        doc_contexts=docs,
        cached_caller=caller,
        agent_config=_config(seq_cover_max_iterations=5),
        logger=logger,
        max_turns=3,
    )
    logger.close()

    assert [rule.retrieval_spec.anchor for rule in result.rules] == ["alpha", "beta"]
    assert result.termination_reason == "full_coverage"
    trajectory_lines = [
        json.loads(line)
        for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
        if line.strip()
    ]
    rejections = [t for t in trajectory_lines if t.get("tool_name") == "generate_rejected_duplicate"]
    assert len(rejections) == 1, f"expected exactly one duplicate rejection, got {len(rejections)}"
    assert rejections[0]["tool_args"]["signature"] == ["regex", "small", "alpha"]


def test_seq_cover_uncovered_window_respected(tmp_path: Path) -> None:
    docs = [
        _doc(f"doc_{idx}", f"token_{idx}", f"token_{idx}")
        for idx in range(5)
    ]
    caller = _FakeCaller(["never"])
    logger = TrajectoryLogger(tmp_path / "trajectory.jsonl")

    run_seq_cover_agent_on_query(
        query_text="find token",
        query_idx=3,
        doc_contexts=docs,
        cached_caller=caller,
        agent_config=_config(
            seq_cover_max_iterations=1,
            seq_cover_uncovered_doc_window=2,
        ),
        logger=logger,
        max_turns=1,
    )
    logger.close()

    first_prompt = caller.prompts[0]
    assert "doc_0" in first_prompt
    assert "doc_1" in first_prompt
    assert "doc_2" not in first_prompt
