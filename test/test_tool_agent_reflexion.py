from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agent.tool_agent.core import AgentConfig
from agent.tool_agent.document import DocumentContext
from agent.tool_agent.logger import TrajectoryLogger
from agent.tool_agent.reflexion_core import run_reflexion_agent_on_query
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
            assert self._anchors, "unexpected structured-response call"
            anchor = self._anchors.pop(0)
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


def test_reflexion_iterates_on_failures(tmp_path: Path) -> None:
    docs = [_doc("doc_a", "alpha", "alpha"), _doc("doc_b", "beta", "beta")]
    caller = _FakeCaller(["alpha", "alpha|beta"])
    logger = TrajectoryLogger(tmp_path / "trajectory.jsonl")

    result = run_reflexion_agent_on_query(
        query_text="find token",
        query_idx=3,
        doc_contexts=docs,
        cached_caller=caller,
        agent_config=_config(reflexion_max_iterations=2),
        logger=logger,
        max_turns=1,
    )
    logger.close()

    assert len(result.rules) == 1
    assert result.rules[0].retrieval_spec.anchor == "alpha|beta"
    trajectory = json.loads((tmp_path / "reflexion_trajectory.json").read_text())
    assert [round_info["best_accuracy"] for round_info in trajectory["rounds"]] == [0.5, 1.0]
    assert "representative_failures" in "\n".join(caller.prompts)


def test_reflexion_force_distinct_action(tmp_path: Path) -> None:
    docs = [_doc("doc_a", "alpha", "alpha"), _doc("doc_b", "beta", "beta")]
    caller = _FakeCaller(["alpha", "alpha", "beta"])
    logger = TrajectoryLogger(tmp_path / "trajectory.jsonl")

    result = run_reflexion_agent_on_query(
        query_text="find token",
        query_idx=3,
        doc_contexts=docs,
        cached_caller=caller,
        agent_config=_config(reflexion_max_iterations=2),
        logger=logger,
        max_turns=2,
    )
    logger.close()

    assert len(result.rules) == 1
    full_log = (tmp_path / "trajectory_full.jsonl").read_text(encoding="utf-8")
    assert "generate_rejected_duplicate" in full_log


def test_failure_memory_ordering_deterministic(tmp_path: Path) -> None:
    docs = [_doc("doc_b", "beta", "beta"), _doc("doc_a", "alpha", "alpha")]
    outputs: list[str] = []

    for run_idx in range(2):
        caller = _FakeCaller(["gamma", "alpha|beta"])
        run_dir = tmp_path / f"run_{run_idx}"
        logger = TrajectoryLogger(run_dir / "trajectory.jsonl")
        run_reflexion_agent_on_query(
            query_text="find token",
            query_idx=3,
            doc_contexts=docs,
            cached_caller=caller,
            agent_config=_config(reflexion_max_iterations=2),
            logger=logger,
            max_turns=1,
        )
        logger.close()
        outputs.append((run_dir / "reflexion_trajectory.json").read_text(encoding="utf-8"))

    assert outputs[0] == outputs[1]


def test_keep_best_only_prunes_pool(tmp_path: Path) -> None:
    docs = [
        _doc("doc_a", "alpha", "alpha"),
        _doc("doc_b", "beta", "beta"),
        _doc("doc_c", "gamma", "gamma"),
    ]
    caller = _FakeCaller(["alpha", "alpha|beta", "alpha|beta|gamma"])
    logger = TrajectoryLogger(tmp_path / "trajectory.jsonl")

    result = run_reflexion_agent_on_query(
        query_text="find token",
        query_idx=3,
        doc_contexts=docs,
        cached_caller=caller,
        agent_config=_config(reflexion_max_iterations=3),
        logger=logger,
        max_turns=1,
    )
    logger.close()

    assert len(result.rules) == 1
    assert result.rules[0].retrieval_spec.anchor == "alpha|beta|gamma"
    trajectory = json.loads((tmp_path / "reflexion_trajectory.json").read_text())
    assert [round_info["pool_size"] for round_info in trajectory["rounds"]] == [1, 1, 1]


def test_reflexion_stops_when_outer_budget_exhausted(tmp_path: Path) -> None:
    docs = [_doc("doc_a", "alpha", "alpha"), _doc("doc_b", "beta", "beta")]
    caller = _FakeCaller(["alpha", "beta"])
    logger = TrajectoryLogger(tmp_path / "trajectory.jsonl")

    result = run_reflexion_agent_on_query(
        query_text="find token",
        query_idx=3,
        doc_contexts=docs,
        cached_caller=caller,
        agent_config=_config(
            budget_usd=0.000001,
            reflexion_max_iterations=2,
            reflexion_per_round_budget_factor=1.0,
        ),
        logger=logger,
        max_turns=1,
    )
    logger.close()

    assert result.termination_reason == "budget"
    assert len([p for p in caller.prompts if "Response Format" in p]) == 1
