from __future__ import annotations

import json
from pathlib import Path

from core.pipeline.e2e_utils.cache import CacheResult
from gt_gen import generator, judge, pipeline


def _make_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "nopv" / "latest"
    raw = root / "raw"
    raw.mkdir(parents=True)
    (root / "queries.json").write_text(
        json.dumps(
            [
                {"text": "Stable question?", "answer_type": "string"},
                {"text": "Disputed question?", "answer_type": "string"},
            ]
        ),
        encoding="utf-8",
    )
    (raw / "doc_a.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    return root


def test_all_self_consistency_pipeline_judges_only_disagreements(
    tmp_path, monkeypatch
):
    root = _make_dataset(tmp_path)
    generation_calls = []

    def fake_generate_ground_truth_for_queries(**kwargs):
        run_root = Path(kwargs["target_dir"])
        run_index = len(generation_calls)
        generation_calls.append(
            {
                "temperature": kwargs["temperature"],
                "model": kwargs["model"],
                "generation_mode": kwargs["generation_mode"],
            }
        )
        output_path = run_root / "ground_truth" / "doc_a.txt_answers.json"
        q2_answer = "variant-b" if run_index else "variant-a"
        generator._atomic_write_json(
            output_path,
            {"1": "stable-answer", "2": q2_answer},
        )
        api_call_id = f"all:doc_a:{','.join(map(str, kwargs['query_indices']))}"
        result_q1 = generator.DocRunResult(
            doc_id="doc_a",
            output_path=output_path,
            status="generated",
            query_idx=1,
            input_tokens=100,
            output_tokens=20,
            cost_usd=0.01,
            latency_ms=2.0,
            api_call_id=api_call_id,
        )
        result_q2 = generator.DocRunResult(
            doc_id="doc_a",
            output_path=output_path,
            status="generated",
            query_idx=2,
            input_tokens=100,
            output_tokens=20,
            cost_usd=0.01,
            latency_ms=2.0,
            api_call_id=api_call_id,
        )
        return generator.BatchGenerationSummary(
            dataset_root=run_root,
            queries=tuple(
                generator.QuerySpec(idx=i, text=f"Q{i}", answer_type="string")
                for i in kwargs["query_indices"]
            ),
            selected_count=1,
            generated_count=2,
            skipped_existing_count=0,
            failed_count=0,
            run_latency_ms=3.0,
            log_path=run_root / "logs" / "fake.log",
            results=(result_q1, result_q2),
        )

    seen_judge = {}

    def fake_judge_document(**kwargs):
        seen_judge["queries"] = [q.idx for q in kwargs["queries"]]
        seen_judge["model"] = kwargs["model"]
        seen_judge["judge_mode"] = kwargs["judge_mode"]
        candidates = kwargs["candidates_by_query"][2]
        return (
            [
                judge.JudgedAnswer(
                    query_idx=2,
                    picked_index=1,
                    picked_source=candidates[1].source,
                    answer="variant-b",
                    reasoning="variant-b matches the document",
                )
            ],
            [
                CacheResult(
                    response="{}",
                    input_tokens=50,
                    output_tokens=10,
                    latency_ms=5.0,
                    cache_hit=False,
                    cost_usd=0.02,
                )
            ],
        )

    monkeypatch.setattr(
        pipeline.generator,
        "generate_ground_truth_for_queries",
        fake_generate_ground_truth_for_queries,
    )
    monkeypatch.setattr(pipeline.judge, "judge_document", fake_judge_document)

    summary = pipeline.run_all_self_consistency_pipeline(
        target_dir=root,
        work_dir=tmp_path / "run",
        query_indices=[1, 2],
        num_doc=1,
        model="gpt-5.4",
        all_runs=3,
        max_tokens=2000,
    )

    assert [call["temperature"] for call in generation_calls] == [0.0, 0.2, 0.2]
    assert all(call["model"] == "gpt-5.4" for call in generation_calls)
    assert all(call["generation_mode"] == "all" for call in generation_calls)
    assert seen_judge == {
        "queries": [2],
        "model": "gpt-5.4",
        "judge_mode": "batched",
    }
    assert summary["accepted_without_judge_count"] == 1
    assert summary["disagreement_cell_count"] == 1
    assert summary["judged_count"] == 1
    assert summary["total_cost_usd"] == 0.05
    assert summary["generation_metrics"]["api_latency_ms"] == 6.0

    final_gt = json.loads(
        (
            Path(summary["final_dataset_root"])
            / "ground_truth"
            / "doc_a.txt_answers.json"
        ).read_text(encoding="utf-8")
    )
    assert final_gt == {"1": "stable-answer", "2": "variant-b"}

    judged = json.loads(
        (
            Path(summary["final_dataset_root"])
            / "judged"
            / "doc_a.judged_answers.json"
        ).read_text(encoding="utf-8")
    )
    assert judged["1"]["decision"] == "consensus"
    assert judged["2"]["decision"] == "judged"
    assert "support" not in judged["2"]
