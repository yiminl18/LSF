from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agent.reflection_agent import runner
from agent.rule_runtime import rule_dispatch
from agent.rule_runtime import holdout
from agent.rule_runtime.data import DocumentSample, get_label_filename
from agent.rules.code_rule_json import CodeRule
from agent.rules.code_rule_sandbox import CodeExecResult
from core.pipeline.e2e_utils.cache import CacheResult


PHONE = "(555) 123-4567"


def _write_reconstructed(path: Path, phone: str = PHONE) -> None:
    path.write_text(
        json.dumps(
            {
                "texts": [
                    {
                        "label": "section_header",
                        "text": "Item 1. Business",
                        "page_no": 1,
                        "text_span": f"Registrant telephone number is {phone}.",
                        "structure": {"level": "H1"},
                    },
                    {
                        "label": "text",
                        "text": f"Registrant telephone number is {phone}.",
                        "page_no": 1,
                        "structure": {"parent_id": 0},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_dataset(tmp_path: Path, doc_count: int) -> Path:
    dataset_root = tmp_path / "dataset"
    processing_dir = dataset_root / "processing"
    label_dir = dataset_root / "label"
    processing_dir.mkdir(parents=True)
    label_dir.mkdir()

    (dataset_root / "queries.txt").write_text(
        "\n".join(["unused q0", "unused q1", "unused q2", "What is the phone number?"])
        + "\n",
        encoding="utf-8",
    )
    labels = []
    for idx in range(doc_count):
        doc_id = f"DOC{idx:03d}"
        _write_reconstructed(processing_dir / f"{doc_id}_reconstructed.json")
        labels.append(
            {
                "doc_name": doc_id,
                "question_idx": 3,
                "ground_truth": PHONE,
                "possible_provenance_nodes": [{"path": "Item 1"}],
            }
        )
    (label_dir / get_label_filename({"dataset": "pdfs"}, 3)).write_text(
        json.dumps({"labels": labels}),
        encoding="utf-8",
    )
    return dataset_root


def _write_config(tmp_path: Path, dataset_root: Path, doc_count: int) -> Path:
    docs = "\n".join(f"      - DOC{idx:03d}" for idx in range(doc_count))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset: pdfs",
                "parser: docling",
                f"dataset_root: {dataset_root}",
                "llm_provider: azure",
                "llm_model: gpt-5.4-mini",
                "max_tokens_output: 512",
                "rule_mode: json_spec_reflect",
                "rule_text_format: normalized",
                "anonymize_doc_ids: false",
                "projected_generation_cost_threshold_usd: 999.0",
                "queries:",
                "  - query_idx: 3",
                "    documents:",
                docs,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _code_text(bundle_index: int) -> str:
    return (
        f"# bundle_{bundle_index}\n"
        "def locate_region(document_text):\n"
        "    marker = \"Registrant telephone number is \"\n"
        "    start = document_text.find(marker)\n"
        "    if start == -1:\n"
        "        return \"\"\n"
        "    end = document_text.find(\"\\n\", start)\n"
        "    if end == -1:\n"
        "        end = len(document_text)\n"
        "    return document_text[start:end]\n"
    )


def _code_response(bundle_index: int) -> str:
    return f"```python\n{_code_text(bundle_index)}```"


def _install_pipeline_fakes(monkeypatch, response_factory=None) -> None:
    code_calls = {"count": 0}

    def fake_call(
        self,
        prompt: str,
        llm_provider: str = "azure",
        max_tokens: int = 800,
        *,
        model: str,
        response_schema: dict | None = None,
        temperature: float = 0,
    ) -> CacheResult:
        idx = code_calls["count"]
        code_calls["count"] += 1
        response = (
            response_factory(idx) if response_factory is not None else _code_response(idx)
        )
        return CacheResult(
            response=response,
            input_tokens=100 + idx,
            output_tokens=20,
            latency_ms=1.0,
            cache_hit=False,
        )

    def fake_execute(code: str, document_text: str) -> CodeExecResult:
        marker = "Registrant telephone number is "
        start = document_text.find(marker)
        if start == -1:
            return CodeExecResult(False, "", "marker not found", 0.1)
        end = document_text.find("\n", start)
        if end == -1:
            end = len(document_text)
        return CodeExecResult(True, document_text[start:end], None, 0.1)

    def fake_score_retrieved_subset(**kwargs):
        return SimpleNamespace(
            generated_answer=PHONE,
            judge_result=True,
            metadata={"generation": {"cost_usd": 0.0}, "judge": {"cost_usd": 0.0}},
        )

    monkeypatch.setattr(runner.CachedLLMCaller, "call", fake_call)
    monkeypatch.setattr(rule_dispatch, "execute_locate_region", fake_execute)
    monkeypatch.setattr(runner, "score_retrieved_subset", fake_score_retrieved_subset)


def _run_code_baseline(
    monkeypatch,
    tmp_path: Path,
    packaging_mode: str,
    doc_count: int,
    response_factory=None,
) -> tuple[Path, dict]:
    dataset_root = _write_dataset(tmp_path, doc_count)
    config_path = _write_config(tmp_path, dataset_root, doc_count)
    output_root = tmp_path / "output"
    _install_pipeline_fakes(monkeypatch, response_factory=response_factory)

    runner.run_baseline_sweep(
        packaging_modes=(packaging_mode,),
        query_indices=(3,),
        config_path=config_path,
        output_root=output_root,
        llm_provider="azure",
        llm_model="gpt-5.4-mini",
        cache_db_path=str(tmp_path / "cache.db"),
        rule_mode="python_code",
    )

    best_rules_path = output_root / "q3" / packaging_mode / "best_rules.json"
    payload = json.loads(best_rules_path.read_text(encoding="utf-8"))
    return output_root, payload


def test_bundle_full_code_phase_a_emits_code_field(monkeypatch, tmp_path: Path) -> None:
    _, payload = _run_code_baseline(
        monkeypatch,
        tmp_path,
        "full_bundle_reference",
        doc_count=1,
    )

    assert payload["rule_mode"] == "python_code"
    assert len(payload["merged_rules"]) == 1
    entry = payload["merged_rules"][0]
    assert entry["rule_kind"] == "code"
    assert "code" in entry
    assert "retrieval_spec" not in entry
    assert set(payload["cross_doc_eval"][0]) >= {
        "rule_index",
        "accuracy",
        "score",
        "success_doc_ids",
    }


def test_bundle_grouped_code_three_groups(monkeypatch, tmp_path: Path) -> None:
    _, payload = _run_code_baseline(
        monkeypatch,
        tmp_path,
        "grouped_433",
        doc_count=10,
    )

    assert len(payload["merged_rules"]) == 3
    assert [entry["rule_kind"] for entry in payload["merged_rules"]] == [
        "code",
        "code",
        "code",
    ]


def test_phase_b_dispatches_to_sandbox(monkeypatch) -> None:
    rule = CodeRule(
        rule_text="find phone with code",
        evidence_basis="test",
        code=_code_text(0),
    )
    doc = DocumentSample(
        doc_id="HOLDCO",
        markdown_text=f"Header\nRegistrant telephone number is {PHONE}.\n",
        ground_truth_answer=PHONE,
        token_count=20,
    )

    def fake_score_retrieved_subset(**kwargs):
        return SimpleNamespace(
            generated_answer=PHONE,
            judge_result=True,
            metadata={"generation": {"cost_usd": 0.0}, "judge": {"cost_usd": 0.0}},
        )

    monkeypatch.setattr(holdout, "score_retrieved_subset", fake_score_retrieved_subset)

    row = holdout.evaluate_rule_on_doc(
        rule,
        0,
        doc,
        3,
        "What is the phone number?",
        cached_caller=object(),
        llm_provider="azure",
        llm_model="gpt-5.4-mini",
        retrieval_too_large_token_threshold=5000,
    )

    assert row["matched"] is True
    assert row["judge_result"] is True
    assert PHONE in row["retrieved_subset_text"]
    assert row["retrieval_spec"]["rule_kind"] == "code"


def test_ast_rejected_logged_no_crash(monkeypatch, tmp_path: Path) -> None:
    def response_factory(idx: int) -> str:
        return (
            "```python\n"
            "import os\n\n"
            "def locate_region(document_text):\n"
            "    return document_text\n"
            "```\n\n"
            f"```python\n{_code_text(idx)}```"
        )

    _, payload = _run_code_baseline(
        monkeypatch,
        tmp_path,
        "full_bundle_reference",
        doc_count=1,
        response_factory=response_factory,
    )

    assert payload["merged_rules"]
    assert payload["rejected_rules"][0]["sandbox_validation_status"] == "rejected_ast"
    assert "forbidden import: os" in payload["rejected_rules"][0]["violations"]
