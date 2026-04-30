from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.rule_runtime.data import get_label_filename
from agent.rules import code_rule_sandbox
from agent.rules.code_rule_json import CodeRule, parse_code_rule_bundle
from agent.tool_agent.code_core import run_code_agent_on_query
from agent.tool_agent.core import AgentConfig
from agent.tool_agent.document import DocumentContext
from agent.tool_agent.logger import TrajectoryLogger
from agent.tool_agent.orchestrator import run_phase_a
from agent.tool_agent.rule_selection import select_best_rules
from agent.tool_agent.tools import ToolRegistry
from core.pipeline.e2e_utils.cache import CacheResult


def _doc(doc_id: str, text: str, ground_truth: str = "alpha") -> DocumentContext:
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


def _valid_code(token: str = "alpha") -> str:
    return (
        "def locate_region(document_text):\n"
        f"    idx = document_text.find({token!r})\n"
        "    if idx == -1:\n"
        "        return \"\"\n"
        f"    return document_text[idx:idx + {len(token)}]\n"
    )


def _invalid_code() -> str:
    return "import os\n\ndef locate_region(document_text):\n    return document_text\n"


def _generate_action(code: str, label: str = "code") -> str:
    return json.dumps(
        {
            "action": "generate",
            "reasoning": "synthetic code rule",
            "tool": None,
            "args": None,
            "rule": json.dumps(
                {
                    "rule_text": f"locate {label}",
                    "evidence_basis": "synthetic fixture",
                    "code": code,
                },
                ensure_ascii=False,
            ),
        },
        ensure_ascii=False,
    )


def _tool_action(tool: str, args: dict[str, Any]) -> str:
    return json.dumps(
        {
            "action": "tool",
            "reasoning": "inspect with tool",
            "tool": tool,
            "args": json.dumps(args, ensure_ascii=False),
            "rule": None,
        },
        ensure_ascii=False,
    )


class _FakeCaller:
    def __init__(self, actions: list[str]) -> None:
        self._actions = list(actions)
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
        assert response_schema is not None, "code agent should only make action calls"
        assert self._actions, "unexpected structured-response call"
        return CacheResult(
            response=self._actions.pop(0),
            input_tokens=10,
            output_tokens=5,
            latency_ms=1.0,
            cache_hit=False,
        )


def _config(**extra: Any) -> AgentConfig:
    cfg = AgentConfig(max_turns_per_query=2, budget_usd=10.0, agent_max_tokens=1000)
    for key, value in extra.items():
        setattr(cfg, key, value)
    return cfg


def _write_reconstructed(path: Path, text: str = "alpha") -> None:
    path.write_text(
        json.dumps(
            {
                "texts": [
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
                        "structure": {"parent_id": 0},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset_root = tmp_path / "dataset"
    processing_dir = dataset_root / "processing"
    label_dir = dataset_root / "label"
    processing_dir.mkdir(parents=True)
    label_dir.mkdir()
    (dataset_root / "queries.txt").write_text(
        "q0\nq1\nq2\nfind alpha\n",
        encoding="utf-8",
    )
    labels = []
    for doc_id in ("doc_a", "doc_b"):
        _write_reconstructed(processing_dir / f"{doc_id}_reconstructed.json")
        labels.append(
            {
                "doc_name": doc_id,
                "question_idx": 3,
                "ground_truth": "alpha",
                "possible_provenance_nodes": [{"path": "Cover"}],
            }
        )
    (label_dir / get_label_filename({"dataset": "pdfs"}, 3)).write_text(
        json.dumps({"labels": labels}),
        encoding="utf-8",
    )
    return dataset_root, processing_dir, label_dir


def _patch_score(monkeypatch) -> None:
    def fake_score_retrieved_subset(**kwargs):
        return SimpleNamespace(
            generated_answer="alpha",
            judge_result="alpha" in kwargs["retrieved_text"],
            metadata={"generation": {"cost_usd": 0.0}, "judge": {"cost_usd": 0.0}},
        )

    monkeypatch.setattr(
        "agent.tool_agent.code_core.score_retrieved_subset",
        fake_score_retrieved_subset,
    )


def _run_phase_a_code(tmp_path: Path, monkeypatch, actions: list[str]) -> Path:
    dataset_root, processing_dir, label_dir = _write_dataset(tmp_path)
    _patch_score(monkeypatch)
    output_dir = tmp_path / "phase_a"
    caller = _FakeCaller(actions)
    run_phase_a(
        query_idx=3,
        doc_ids=["doc_a", "doc_b"],
        processing_dir=processing_dir,
        label_dir=label_dir,
        dataset_root=str(dataset_root),
        dataset_name="pdfs",
        truncate_before=None,
        cached_caller=caller,
        agent_config=_config(max_turns_per_query=len(actions)),
        output_dir=output_dir,
        mode="code",
    )
    return output_dir


def test_e2e_one_valid_one_rejected_code_rule(tmp_path: Path) -> None:
    docs = [_doc("doc_a", "alpha"), _doc("doc_b", "alpha")]
    caller = _FakeCaller([
        _generate_action(_invalid_code(), "bad"),
        _generate_action(_valid_code(), "alpha"),
    ])
    logger = TrajectoryLogger(tmp_path / "trajectory.jsonl")

    result = run_code_agent_on_query(
        query_text="find alpha",
        query_idx=3,
        doc_contexts=docs,
        cached_caller=caller,
        agent_config=_config(max_turns_per_query=2),
        logger=logger,
    )
    logger.close()

    assert len(result.rules) == 1
    assert result.trajectory[0]["code_rule_rejections"]["ast"] == 1
    assert "generate_rejected_ast" in (tmp_path / "trajectory_full.jsonl").read_text()


def test_try_code_rule_returns_sandbox_output(tmp_path: Path) -> None:
    docs = [_doc("doc_a", "alpha"), _doc("doc_b", "beta", "beta")]
    caller = _FakeCaller([
        _tool_action("try_code_rule", {"code": _valid_code(), "doc_ids": ["doc_a"]}),
        _generate_action(_valid_code(), "alpha"),
    ])
    logger = TrajectoryLogger(tmp_path / "trajectory.jsonl")

    result = run_code_agent_on_query(
        query_text="find alpha",
        query_idx=3,
        doc_contexts=docs,
        cached_caller=caller,
        agent_config=_config(max_turns_per_query=2),
        logger=logger,
    )
    logger.close()

    assert len(result.rules) == 1
    assert "alpha" in caller.prompts[1]


def test_try_code_rule_rejection_counted_in_agent_metrics(tmp_path: Path) -> None:
    docs = [_doc("doc_a", "alpha")]
    caller = _FakeCaller([
        _tool_action("try_code_rule", {"code": _invalid_code(), "doc_ids": ["doc_a"]}),
        _generate_action(_valid_code(), "alpha"),
    ])
    logger = TrajectoryLogger(tmp_path / "trajectory.jsonl")

    result = run_code_agent_on_query(
        query_text="find alpha",
        query_idx=3,
        doc_contexts=docs,
        cached_caller=caller,
        agent_config=_config(max_turns_per_query=2),
        logger=logger,
    )
    logger.close()

    assert len(result.rules) == 1
    assert result.trajectory[0]["code_rule_rejections"] == {
        "ast": 1,
        "timeout": 0,
        "runtime_error": 0,
    }


def test_determinism_under_cache(tmp_path: Path, monkeypatch) -> None:
    outputs: list[tuple[str, str]] = []
    for run_idx in range(2):
        output_dir = _run_phase_a_code(
            tmp_path / f"run_{run_idx}",
            monkeypatch,
            [_generate_action(_valid_code(), "alpha")],
        )
        outputs.append(
            (
                (output_dir / "best_rules.json").read_text(encoding="utf-8"),
                (output_dir / "phase_a_docs.json").read_text(encoding="utf-8"),
            )
        )

    assert outputs[0] == outputs[1]


def test_ast_rejected_recorded_as_metric_not_failure(tmp_path: Path, monkeypatch) -> None:
    output_dir = _run_phase_a_code(
        tmp_path,
        monkeypatch,
        [
            _generate_action(_invalid_code(), "bad"),
            _generate_action(_valid_code(), "alpha"),
        ],
    )

    report = json.loads((output_dir / "phase_a_report.json").read_text(encoding="utf-8"))
    assert report["code_rule_rejections"] == {
        "ast": 1,
        "timeout": 0,
        "runtime_error": 0,
    }
    assert report["rule_count"] == 1


def test_ast_rejection_deterministic_under_cache() -> None:
    raw = (
        f"```python\n{_valid_code('alpha')}```\n"
        f"```python\n{_invalid_code()}```\n"
        f"```python\n{_valid_code('beta')}```\n"
    )

    first = parse_code_rule_bundle(raw, query_idx=3)
    second = parse_code_rule_bundle(raw, query_idx=3)

    assert [rule.code for rule in first.rules] == [rule.code for rule in second.rules]
    assert [rule.code for rule in first.rules] == [_valid_code("alpha").strip(), _valid_code("beta").strip()]


def test_phase_a_docs_shape_matches_orchestrator(tmp_path: Path, monkeypatch) -> None:
    output_dir = _run_phase_a_code(
        tmp_path,
        monkeypatch,
        [_generate_action(_valid_code(), "alpha")],
    )

    payload = json.loads((output_dir / "phase_a_docs.json").read_text(encoding="utf-8"))

    assert list(payload) == ["query_idx", "excluded_doc_ids", "processed_doc_ids", "count"]
    assert payload["excluded_doc_ids"] == ["doc_a", "doc_b"]
    assert payload["processed_doc_ids"] == ["doc_a", "doc_b"]


def test_select_best_rules_consumes_code_cross_doc_eval() -> None:
    rule = CodeRule("locate alpha", "test", _valid_code())
    selected = select_best_rules(
        [
            {
                "rule_index": 0,
                "rule_kind": "code",
                "accuracy": 1.0,
                "score": 1.0,
                "success_doc_ids": ["doc_a"],
                "skipped_due_to_budget": False,
            }
        ],
        [rule],
        max_rules=1,
    )

    assert selected == [rule]


def test_try_code_rule_rejects_forbidden_builtin_via_validate_code_ast() -> None:
    registry = ToolRegistry(_doc("doc_a", "alpha"))

    for name in sorted(code_rule_sandbox._FORBIDDEN_NAMES):
        code = (
            "def locate_region(document_text):\n"
            f"    {name}\n"
            "    return document_text\n"
        )
        result = registry.dispatch("try_code_rule", {"code": code, "doc_ids": ["doc_a"]})

        assert result.success is True
        error = result.data["per_doc"][0]["error_or_none"]
        assert error is not None
        assert "AST violations:" in error
        assert name in error
        assert result.data["per_doc"][0]["error_kind"] == "ast"


def test_low_accuracy_code_rule_filtered() -> None:
    rule = CodeRule("locate alpha", "test", _valid_code())
    selected = select_best_rules(
        [
            {
                "rule_index": 0,
                "rule_kind": "code",
                "accuracy": 0.0,
                "score": 0.0,
                "success_doc_ids": [],
                "skipped_due_to_budget": False,
            }
        ],
        [rule],
        max_rules=1,
        allow_no_score_first_three_fallback=False,
    )

    assert selected == []
