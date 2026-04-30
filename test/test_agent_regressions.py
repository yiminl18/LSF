from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agent.rule_runtime import deploy, holdout
from agent.rule_runtime.data import DocumentSample, get_label_filename
from agent.rules.code_rule_json import CodeRule
from agent.rules.code_rule_sandbox import execute_locate_region
from agent.rules.range_rule_json import RangeRule, RetrievalSpec
from agent.tool_agent import orchestrator
from agent.tool_agent.document import DocumentContext, load_document_context


def _best_rule_entry(**extra: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "rule_text": "find needle",
        "evidence_basis": "test",
        "retrieval_spec": {
            "mode": "regex",
            "anchor": "needle",
            "anchor_b": None,
            "page_idx": None,
            "max_chars": 64,
            "boundary_context_chars": 0,
        },
    }
    entry.update(extra)
    return entry


def test_load_frozen_rules_excludes_bundle_and_summary_docs(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    package_dir = output_root / "q3" / "grouped_433"
    package_dir.mkdir(parents=True)
    (package_dir / "best_rules.json").write_text(
        json.dumps(
            {
                "merged_rules": [
                    _best_rule_entry(
                        primary_doc_ids=["primary_doc"],
                        source_bundle_doc_ids_list=[
                            ["bundle_doc_a", "bundle_doc_b"],
                            ["bundle_doc_c"],
                        ],
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    (package_dir / "summary.json").write_text(
        json.dumps({"selected_doc_ids": ["summary_doc", "bundle_doc_b"]}),
        encoding="utf-8",
    )

    rules, sampled_doc_ids = holdout.load_frozen_rules(3, "grouped_433", output_root)

    assert len(rules) == 1
    assert sampled_doc_ids == {
        "primary_doc",
        "bundle_doc_a",
        "bundle_doc_b",
        "bundle_doc_c",
        "summary_doc",
    }


def test_deploy_sampled_doc_ids_include_bundle_and_summary_docs() -> None:
    payload = {
        "merged_rules": [
            _best_rule_entry(
                primary_doc_ids=["primary_doc"],
                source_bundle_doc_ids_list=[["bundle_doc"]],
            )
        ]
    }
    config = {"queries": [{"query_idx": 3, "documents": ["fallback_doc"]}]}

    sampled_doc_ids = deploy._sampled_doc_ids_from_payload(
        payload,
        config,
        3,
        sampled_summary={"selected_doc_ids": ["summary_doc"]},
    )

    assert sampled_doc_ids == {"primary_doc", "bundle_doc", "summary_doc"}
    assert deploy._sampled_doc_ids_from_payload({}, config, 3) == {"fallback_doc"}


def test_deploy_cascade_dispatches_code_rule_to_sandbox(monkeypatch) -> None:
    phone = "(555) 123-4567"
    expected_region = f"Registrant telephone number is {phone}."
    rule = CodeRule(
        rule_text="find phone with code",
        evidence_basis="test",
        code=(
            "def locate_region(document_text):\n"
            "    marker = 'Registrant telephone number is '\n"
            "    start = document_text.find(marker)\n"
            "    if start == -1:\n"
            "        return ''\n"
            "    end = document_text.find('\\n', start)\n"
            "    if end == -1:\n"
            "        end = len(document_text)\n"
            "    return document_text[start:end]\n"
        ),
    )
    doc = DocumentSample(
        doc_id="HOLDCO",
        markdown_text=f"Header\nRegistrant telephone number is {phone}.\n",
        ground_truth_answer=phone,
        token_count=20,
    )

    def fake_generate_answer_from_text(**kwargs):
        assert kwargs["retrieved_text"] == expected_region
        return SimpleNamespace(answer=phone, cost_usd=0.0)

    monkeypatch.setattr(
        deploy,
        "generate_answer_from_text",
        fake_generate_answer_from_text,
    )
    monkeypatch.setattr(
        deploy,
        "score_generated_answer",
        lambda **kwargs: SimpleNamespace(
            judge_result=True,
            metadata={"judge": {"cost_usd": 0.0}},
        ),
    )

    rows = deploy.evaluate_cascade(
        rules=[rule],
        sampled_eval={0: {"accuracy": 1.0, "score": 1.0}},
        holdout_docs=[doc],
        query_idx=3,
        query_text="What is the phone number?",
        cached_caller=object(),
        llm_provider="azure",
        llm_model="gpt-5.4-mini",
        retrieval_too_large_token_threshold=5000,
    )

    assert len(rows) == 1
    assert rows[0]["rules_used"] == [0]
    assert rows[0]["judge_result"] is True
    assert rows[0]["blocker"] is None
    assert rows[0]["retrieved_subset_text"] == expected_region


def test_holdout_cli_cache_default_is_imported() -> None:
    assert holdout.DEFAULT_CACHE_DB_PATH


def test_failed_code_rule_returns_no_fallback_region() -> None:
    document_text = "answer appears near the top of the document"
    result = execute_locate_region(
        "import os\n\n"
        "def locate_region(document_text):\n"
        "    return document_text[:2000]\n",
        document_text,
    )

    assert result.success is False
    assert result.returned_region == ""
    assert result.error is not None
    assert result.error_kind == "ast"
    assert "forbidden import" in result.error


def test_cross_doc_budget_truncation_scores_against_full_corpus(monkeypatch) -> None:
    rule = RangeRule(
        rule_text="find needle",
        evidence_basis="test",
        retrieval_spec=RetrievalSpec(
            mode="regex",
            anchor="needle",
            anchor_b=None,
            page_idx=None,
            max_chars=64,
        ),
    )
    doc_contexts = [
        DocumentContext("doc_a", 3, "needle answer a", "answer"),
        DocumentContext("doc_b", 3, "needle answer b", "answer"),
    ]

    def fake_score_retrieved_subset(**kwargs):
        return SimpleNamespace(
            judge_result=True,
            metadata={
                "generation": {"cost_usd": 0.02},
                "judge": {"cost_usd": 0.0},
            },
        )

    monkeypatch.setattr(orchestrator, "score_retrieved_subset", fake_score_retrieved_subset)

    stats = orchestrator._cross_doc_evaluate(
        [rule],
        doc_contexts,
        query_text="question",
        cached_caller=object(),
        llm_provider="provider",
        llm_model="model",
        remaining_budget=0.01,
    )

    assert stats[0]["evaluated_docs"] == 1
    assert stats[0]["total_docs"] == 2
    assert stats[0]["budget_truncated"] is True
    assert stats[0]["coverage"] == 0.5
    assert stats[0]["accuracy"] == 0.5
    assert stats[0]["score"] == 0.25


def test_load_document_context_uses_dataset_specific_label_filename(tmp_path: Path) -> None:
    processing_dir = tmp_path / "processing"
    label_dir = tmp_path / "label"
    processing_dir.mkdir()
    label_dir.mkdir()

    (processing_dir / "DOC1_reconstructed.json").write_text(
        json.dumps(
            {
                "texts": [
                    {
                        "label": "text",
                        "text": "The answer is alpha.",
                        "page_no": 1,
                        "structure": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (label_dir / get_label_filename({"dataset": "sci-docs"}, 7)).write_text(
        json.dumps(
            {
                "labels": [
                    {
                        "doc_name": "DOC1",
                        "question_idx": 7,
                        "ground_truth": "alpha",
                        "possible_provenance_nodes": [{"path": "Abstract"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    ctx = load_document_context(
        "DOC1",
        7,
        processing_dir,
        label_dir,
        dataset_name="sci-docs",
    )

    assert ctx.ground_truth == "alpha"
