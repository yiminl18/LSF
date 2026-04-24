from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.tool_agent import cli as cli_module
from core.pipeline.e2e_utils.cache import CacheResult


PHONE = "(555) 123-4567"


def _write_reconstructed(path: Path, phone: str) -> None:
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


def test_tool_agent_phase_a_phase_b_cascade_e2e(monkeypatch, tmp_path: Path) -> None:
    source_dataset = tmp_path / "source_dataset"
    processing_dir = source_dataset / "processing"
    label_dir = source_dataset / "label"
    processing_dir.mkdir(parents=True)
    label_dir.mkdir()

    (source_dataset / "queries.txt").write_text(
        "\n".join(
            [
                "unused q0",
                "unused q1",
                "unused q2",
                "What is the registrant's telephone number?",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_reconstructed(processing_dir / "TRAINCO_2024_10K_reconstructed.json", PHONE)
    _write_reconstructed(processing_dir / "HOLDCO_2024_10K_reconstructed.json", PHONE)
    (label_dir / "10k_q3_reconstructed_labels.json").write_text(
        json.dumps(
            {
                "labels": [
                    {
                        "doc_name": "TRAINCO_2024_10K",
                        "question_idx": 3,
                        "ground_truth": PHONE,
                        "possible_provenance_nodes": [{"path": "Item 1"}],
                    },
                    {
                        "doc_name": "HOLDCO_2024_10K",
                        "question_idx": 3,
                        "ground_truth": PHONE,
                        "possible_provenance_nodes": [{"path": "Item 1"}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    latest_link = tmp_path / "datasets" / "pdfs" / "latest"
    latest_link.parent.mkdir(parents=True)
    latest_link.symlink_to(source_dataset, target_is_directory=True)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset: pdfs",
                "parser: docling",
                f"dataset_root: {latest_link}",
                "llm_provider: azure",
                "llm_model: gpt-5.4-mini",
                "queries:",
                "  - query_idx: 3",
                "    documents:",
                "      - TRAINCO_2024_10K",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "output"
    monkeypatch.setattr(
        cli_module,
        "parse_args",
        lambda: argparse.Namespace(
            config=config_path,
            queries="3",
            phase="both",
            max_docs=1,
            max_holdout_docs=1,
            holdout_seed=42,
            agent_provider="azure",
            agent_model="gpt-5.4-mini",
            eval_provider=None,
            eval_model=None,
            max_turns=3,
            budget=2.0,
            output_root=output_root,
            experiment_name="exp",
            dry_run=False,
            multipath_n=2,
            partition_seed=42,
            mode="single_shot",
        ),
    )

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
        if response_schema is not None:
            rule = {
                "rule_text": "match the registrant telephone number",
                "evidence_basis": "phone numbers are stable SEC cover-page values",
                "retrieval_spec": {
                    "mode": "regex",
                    "anchor": r"\(\d{3}\) \d{3}-\d{4}",
                    "anchor_b": None,
                    "page_idx": None,
                    "max_chars": 32,
                    "boundary_context_chars": 0,
                },
                "answer_hint_pattern": r"\(\d{3}\) \d{3}-\d{4}",
            }
            action = {
                "action": "submit",
                "reasoning": "use the phone value format",
                "tool": None,
                "args": None,
                "rules": json.dumps([rule]),
            }
            response = json.dumps(action)
        elif 'Return ONLY "True" or "False"' in prompt:
            response = "True"
        else:
            response = PHONE
        return CacheResult(
            response=response,
            input_tokens=100,
            output_tokens=10,
            latency_ms=1.0,
            cache_hit=False,
        )

    monkeypatch.setattr(cli_module.CachedLLMCaller, "call", fake_call)

    cli_module.main()

    phase_a_dir = output_root / "q3" / "exp" / "phase_a"
    best_rules_path = phase_a_dir / "best_rules.json"
    assert best_rules_path.exists()
    best_rules = json.loads(best_rules_path.read_text(encoding="utf-8"))
    assert len(best_rules["merged_rules"]) >= 1
    assert best_rules["merged_rules"][0]["retrieval_spec"]["mode"] == "regex"
    removed_key = "_".join(["gen", "prompt", "template"])
    removed_artifact = "_".join(["meta", "gen"]) + ".json"
    assert removed_key not in best_rules
    assert not (phase_a_dir / removed_artifact).exists()

    phase_b_dir = output_root / "q3" / "exp" / "phase_b"
    report_path = phase_b_dir / "holdout_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["cascade_summary"]["policy"] == "cascade"
    assert report["cascade_summary"]["total_docs"] == 1

    per_rule_rows = [
        json.loads(line)
        for line in (phase_b_dir / "holdout_eval_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(per_rule_rows) == 1
    assert per_rule_rows[0]["retrieved_subset_text"] == PHONE

    union_rows = [
        json.loads(line)
        for line in (phase_b_dir / "holdout_union_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(union_rows) == 1
    assert union_rows[0]["retrieved_subset_text"] == PHONE

    cascade_rows = [
        json.loads(line)
        for line in (phase_b_dir / "holdout_cascade_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(cascade_rows) == 1
    assert cascade_rows[0]["judge_result"] is True
    assert cascade_rows[0]["retrieved_subset_text"] == PHONE
