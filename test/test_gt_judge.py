from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.pipeline.e2e_utils.cache import CacheResult
from gt_gen import generator, judge


@pytest.fixture(autouse=True)
def _isolate_default_log_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _make_nopv_dataset(tmp_path: Path, names=("doc_a",)) -> Path:
    root = tmp_path / "nopv" / "latest"
    raw = root / "raw"
    raw.mkdir(parents=True)
    (root / "queries.json").write_text(
        json.dumps(
            [
                {"text": "Q1 text", "answer_type": "string"},
                {"text": "Q2 text", "answer_type": "integer"},
            ]
        ),
        encoding="utf-8",
    )
    for name in names:
        (raw / f"{name}.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    return root


def _write_manifest(
    manifest_path: Path,
    root: Path,
    *,
    doc_name: str = "doc_a",
    candidates: dict | None = None,
) -> Path:
    pdf_path = root / "raw" / f"{doc_name}.pdf"
    candidates = candidates or {
        "1": [
            {"source": "single@gpt-5.4", "answer": "alpha", "reasoning": "r1", "support": "s1"},
            {"source": "all@gpt-5.4", "answer": "beta", "reasoning": "r2", "support": "s2"},
        ]
    }
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "nopv",
                "documents": [
                    {
                        "doc_id": doc_name,
                        "pdf_path": str(pdf_path),
                        "candidates": candidates,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def test_judge_prompt_template_renders_candidates():
    query = generator.QuerySpec(idx=8, text="Q8 text", answer_type="integer")
    candidates = [
        judge.Candidate(source="single", answer=2, reasoning="A and B are mandatory", support="..."),
        judge.Candidate(source="all", answer=3, reasoning="counted item C", support="..."),
    ]
    template = generator._load_gt_prompt_template("nopv")

    prompt = judge.build_judge_prompt(
        query=query,
        candidates=candidates,
        doc_id="doc_a",
        prompt_template=template,
    )

    assert "{{" not in prompt
    assert "NOPV labeling rules:" in prompt
    assert "Candidate 0 (source=single)" in prompt
    assert "Candidate 1 (source=all)" in prompt
    assert '"picked_index"' in prompt
    assert "Question index: 8" in prompt
    assert prompt.index("Candidate 0") < prompt.index("Candidate 1")


def test_judge_text_prompt_includes_document_text_before_question():
    query = generator.QuerySpec(idx=1, text="Q1", answer_type="string")
    candidates = [judge.Candidate(source="single", answer="x")]
    template = generator._load_gt_prompt_template("nopv")

    prompt = judge.build_judge_text_prompt(
        query=query,
        candidates=candidates,
        doc_id="doc_a",
        document_text="[DOCUMENT TEXT START]\nshared document text",
        prompt_template=template,
    )

    assert prompt.index("shared document text") < prompt.index("[QUESTION]")
    assert "Candidates:" in prompt
    assert "Candidate 0 (source=single)" in prompt


def test_judge_all_prompt_lists_every_query():
    queries = [
        generator.QuerySpec(idx=1, text="Q1", answer_type="string"),
        generator.QuerySpec(idx=2, text="Q2", answer_type="integer"),
        generator.QuerySpec(idx=8, text="Q8", answer_type="integer"),
    ]
    candidates_by_query = {
        1: [judge.Candidate(source="single", answer="a")],
        2: [judge.Candidate(source="single", answer=2)],
        8: [
            judge.Candidate(source="single", answer=2),
            judge.Candidate(source="all", answer=3),
        ],
    }
    template = generator._load_gt_prompt_template("nopv")

    prompt = judge.build_judge_all_prompt(
        queries=queries,
        candidates_by_query=candidates_by_query,
        doc_id="doc_a",
        prompt_template=template,
    )

    assert "{{" not in prompt
    for idx in (1, 2, 8):
        assert f"Question index: {idx}" in prompt
    assert "Candidate 1 (source=all)" in prompt  # Q8's second candidate


def test_judge_response_schema_shape():
    schema = judge._judge_response_schema()
    assert schema["required"] == [
        "reasoning",
        "picked_index",
        "picked_source",
        "answer",
    ]
    assert schema["properties"]["picked_index"]["type"] == "integer"
    assert schema["properties"]["picked_source"]["type"] == "string"
    assert "support" not in schema["properties"]

    multi_schema = judge._judges_response_schema(
        [
            generator.QuerySpec(idx=1, text="Q1", answer_type="string"),
            generator.QuerySpec(idx=2, text="Q2", answer_type="integer"),
        ]
    )
    item_schema = multi_schema["properties"]["judgments"]["items"]
    assert item_schema["required"] == [
        "query_idx",
        "reasoning",
        "picked_index",
        "picked_source",
        "answer",
    ]
    assert item_schema["properties"]["query_idx"]["enum"] == [1, 2]
    assert "support" not in item_schema["properties"]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def test_parse_judge_response_returns_picked_fields():
    candidates = [
        judge.Candidate(source="single", answer="a"),
        judge.Candidate(source="all", answer="b"),
    ]
    raw = json.dumps(
        {
            "reasoning": "...",
            "picked_index": 1,
            "picked_source": "all",
            "answer": "b",
            "support": "page 2",
        }
    )

    picked_index, picked_source, answer, reasoning = judge.parse_judge_response(
        raw, candidates
    )

    assert picked_index == 1
    assert picked_source == "all"
    assert answer == "b"
    assert reasoning == "..."


def test_parse_judge_response_normalizes_out_of_range_pick():
    candidates = [judge.Candidate(source="single", answer="a")]
    raw = json.dumps(
        {
            "reasoning": "all wrong",
            "picked_index": 7,
            "picked_source": "single",
            "answer": "corrected",
            "support": "page 1",
        }
    )

    picked_index, picked_source, answer, *_ = judge.parse_judge_response(
        raw, candidates
    )

    assert picked_index == -1
    assert picked_source == "judge"
    assert answer == "corrected"


def test_parse_judge_response_corrects_source_label():
    candidates = [
        judge.Candidate(source="single@gpt-5.4", answer="a"),
        judge.Candidate(source="all@gpt-5.4", answer="b"),
    ]
    raw = json.dumps(
        {
            "reasoning": "ok",
            "picked_index": 0,
            "picked_source": "WRONG_LABEL",
            "answer": "a",
            "support": "p1",
        }
    )

    picked_index, picked_source, *_ = judge.parse_judge_response(raw, candidates)

    assert picked_index == 0
    assert picked_source == "single@gpt-5.4"


def test_parse_judges_response_rejects_missing_duplicate_and_unexpected():
    queries = [
        generator.QuerySpec(idx=1, text="Q1", answer_type="string"),
        generator.QuerySpec(idx=2, text="Q2", answer_type="string"),
    ]
    cbq = {1: [], 2: []}

    with pytest.raises(ValueError, match="Missing answers"):
        judge.parse_judges_response(
            json.dumps(
                {
                    "judgments": [
                        {
                            "query_idx": 1,
                            "reasoning": "r",
                            "picked_index": -1,
                            "picked_source": "judge",
                            "answer": "a",
                            "support": "s",
                        }
                    ]
                }
            ),
            queries,
            cbq,
        )
    with pytest.raises(ValueError, match="Duplicate"):
        judge.parse_judges_response(
            json.dumps(
                {
                    "judgments": [
                        {
                            "query_idx": 1,
                            "reasoning": "r",
                            "picked_index": -1,
                            "picked_source": "judge",
                            "answer": "a",
                            "support": "s",
                        },
                        {
                            "query_idx": 1,
                            "reasoning": "r",
                            "picked_index": -1,
                            "picked_source": "judge",
                            "answer": "b",
                            "support": "s",
                        },
                    ]
                }
            ),
            [queries[0]],
            {1: []},
        )
    with pytest.raises(ValueError, match="Unexpected"):
        judge.parse_judges_response(
            json.dumps(
                {
                    "judgments": [
                        {
                            "query_idx": 99,
                            "reasoning": "r",
                            "picked_index": -1,
                            "picked_source": "judge",
                            "answer": "a",
                            "support": "s",
                        }
                    ]
                }
            ),
            [queries[0]],
            {1: []},
        )


# ---------------------------------------------------------------------------
# judge_document and end-to-end driver
# ---------------------------------------------------------------------------


def test_judge_document_per_query_picks_candidate(tmp_path, monkeypatch):
    root = _make_nopv_dataset(tmp_path)
    monkeypatch.setattr(
        judge,
        "_extract_pdf_text_for_prompt",
        lambda pdf_path: "[DOCUMENT TEXT START]\ndoc",
    )
    seen_prompts = []

    def fake_call(self, prompt, **kwargs):
        seen_prompts.append(prompt)
        return CacheResult(
            response=json.dumps(
                {
                    "reasoning": "candidate 0 reasoning checks out",
                    "picked_index": 0,
                    "picked_source": "single",
                    "answer": "alpha",
                    "support": "page 1",
                }
            ),
            input_tokens=10,
            output_tokens=5,
            latency_ms=1.0,
            cache_hit=False,
            cost_usd=0.0001,
        )

    monkeypatch.setattr(generator.AzureResponsesTextCacheCaller, "call", fake_call)

    query = generator.QuerySpec(idx=1, text="Q1", answer_type="string")
    candidates = [
        judge.Candidate(source="single", answer="alpha", reasoning="r1", support="s1"),
        judge.Candidate(source="all", answer="beta", reasoning="r2", support="s2"),
    ]

    judged_answers, cache_results = judge.judge_document(
        pdf_path=root / "raw" / "doc_a.pdf",
        queries=[query],
        candidates_by_query={1: candidates},
        cache_db=str(tmp_path / "judge.db"),
    )

    assert len(judged_answers) == 1
    assert judged_answers[0].query_idx == 1
    assert judged_answers[0].picked_index == 0
    assert judged_answers[0].picked_source == "single"
    assert judged_answers[0].answer == "alpha"
    assert len(cache_results) == 1
    assert "Candidate 0 (source=single)" in seen_prompts[0]
    assert "Candidate 1 (source=all)" in seen_prompts[0]


def test_judge_document_batched_one_call_for_all_queries(tmp_path, monkeypatch):
    root = _make_nopv_dataset(tmp_path)
    monkeypatch.setattr(
        judge,
        "_extract_pdf_text_for_prompt",
        lambda pdf_path: "[DOCUMENT TEXT START]\ndoc",
    )
    call_count = {"n": 0}

    def fake_call(self, prompt, **kwargs):
        call_count["n"] += 1
        return CacheResult(
            response=json.dumps(
                {
                    "judgments": [
                        {
                            "query_idx": 1,
                            "reasoning": "r",
                            "picked_index": 0,
                            "picked_source": "single",
                            "answer": "alpha",
                            "support": "p1",
                        },
                        {
                            "query_idx": 2,
                            "reasoning": "r",
                            "picked_index": -1,
                            "picked_source": "judge",
                            "answer": 42,
                            "support": "p2",
                        },
                    ]
                }
            ),
            input_tokens=20,
            output_tokens=10,
            latency_ms=1.0,
            cache_hit=False,
            cost_usd=0.0002,
        )

    monkeypatch.setattr(generator.AzureResponsesTextCacheCaller, "call", fake_call)

    queries = [
        generator.QuerySpec(idx=1, text="Q1", answer_type="string"),
        generator.QuerySpec(idx=2, text="Q2", answer_type="integer"),
    ]
    candidates_by_query = {
        1: [judge.Candidate(source="single", answer="alpha")],
        2: [judge.Candidate(source="single", answer=2)],
    }

    judged_answers, cache_results = judge.judge_document(
        pdf_path=root / "raw" / "doc_a.pdf",
        queries=queries,
        candidates_by_query=candidates_by_query,
        judge_mode="batched",
        cache_db=str(tmp_path / "judge.db"),
    )

    assert call_count["n"] == 1
    assert len(cache_results) == 1
    assert [j.query_idx for j in judged_answers] == [1, 2]
    assert judged_answers[1].picked_index == -1
    assert judged_answers[1].picked_source == "judge"
    assert judged_answers[1].answer == 42


def test_judge_document_correction_when_no_candidate_matches(tmp_path, monkeypatch):
    root = _make_nopv_dataset(tmp_path)
    monkeypatch.setattr(
        judge, "_extract_pdf_text_for_prompt", lambda pdf_path: "[DOC]\nbody"
    )

    def fake_call(self, prompt, **kwargs):
        return CacheResult(
            response=json.dumps(
                {
                    "reasoning": "all wrong",
                    "picked_index": -1,
                    "picked_source": "judge",
                    "answer": "corrected",
                    "support": "page 9",
                }
            ),
            input_tokens=1,
            output_tokens=1,
            latency_ms=0.5,
            cache_hit=False,
        )

    monkeypatch.setattr(generator.AzureResponsesTextCacheCaller, "call", fake_call)

    judged_answers, _ = judge.judge_document(
        pdf_path=root / "raw" / "doc_a.pdf",
        queries=[generator.QuerySpec(idx=1, text="Q1", answer_type="string")],
        candidates_by_query={
            1: [judge.Candidate(source="single", answer="alpha")]
        },
        cache_db=str(tmp_path / "judge.db"),
    )

    assert judged_answers[0].picked_index == -1
    assert judged_answers[0].picked_source == "judge"
    assert judged_answers[0].answer == "corrected"


def test_judge_writes_output_file_without_touching_ground_truth(tmp_path, monkeypatch):
    root = _make_nopv_dataset(tmp_path)
    gt_dir = root / "ground_truth"
    gt_dir.mkdir()
    gt_file = gt_dir / "doc_a.txt_answers.json"
    gt_file.write_text(json.dumps({"1": "untouched"}), encoding="utf-8")

    monkeypatch.setattr(
        judge, "_extract_pdf_text_for_prompt", lambda pdf_path: "[DOC]\nbody"
    )
    monkeypatch.setattr(
        generator.AzureResponsesTextCacheCaller,
        "call",
        lambda self, prompt, **kwargs: CacheResult(
            response=json.dumps(
                {
                    "reasoning": "r",
                    "picked_index": 0,
                    "picked_source": "single@gpt-5.4",
                    "answer": "alpha",
                    "support": "page 1",
                }
            ),
            input_tokens=1,
            output_tokens=1,
            latency_ms=0.5,
            cache_hit=False,
        ),
    )

    manifest_path = _write_manifest(tmp_path / "manifest.json", root)

    summary = judge.judge_ground_truth_from_manifest(
        target_dir=root,
        manifest_path=manifest_path,
        query_indices=[1],
        cache_db=str(tmp_path / "judge.db"),
    )

    output_path = root / "judged" / "doc_a.judged_answers.json"
    assert output_path.exists()
    judged = json.loads(output_path.read_text(encoding="utf-8"))
    assert judged["1"]["answer"] == "alpha"
    assert judged["1"]["picked_index"] == 0
    assert judged["1"]["picked_source"] == "single@gpt-5.4"

    assert json.loads(gt_file.read_text(encoding="utf-8")) == {"1": "untouched"}
    assert summary.judged_count == 1
    assert summary.skipped_existing_count == 0


def test_judge_skips_existing_judged_answers(tmp_path, monkeypatch):
    root = _make_nopv_dataset(tmp_path)
    judged_dir = root / "judged"
    judged_dir.mkdir()
    (judged_dir / "doc_a.judged_answers.json").write_text(
        json.dumps(
            {
                "1": {
                    "answer": "pre-existing",
                    "picked_index": 0,
                    "picked_source": "single",
                    "reasoning": "r",
                    "support": "s",
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        judge, "_extract_pdf_text_for_prompt", lambda pdf_path: "[DOC]\nbody"
    )
    call_count = {"n": 0}

    def fake_call(self, prompt, **kwargs):
        call_count["n"] += 1
        return CacheResult(
            response=json.dumps(
                {
                    "reasoning": "r",
                    "picked_index": 0,
                    "picked_source": "single",
                    "answer": 7,
                    "support": "page 2",
                }
            ),
            input_tokens=1,
            output_tokens=1,
            latency_ms=0.5,
            cache_hit=False,
        )

    monkeypatch.setattr(generator.AzureResponsesTextCacheCaller, "call", fake_call)

    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        root,
        candidates={
            "1": [{"source": "single", "answer": "alpha"}],
            "2": [{"source": "single", "answer": 7}],
        },
    )

    summary = judge.judge_ground_truth_from_manifest(
        target_dir=root,
        manifest_path=manifest_path,
        query_indices=[1, 2],
        cache_db=str(tmp_path / "judge.db"),
    )

    assert call_count["n"] == 1
    assert summary.judged_count == 1
    assert summary.skipped_existing_count == 1

    judged = json.loads(
        (root / "judged" / "doc_a.judged_answers.json").read_text(encoding="utf-8")
    )
    assert judged["1"]["answer"] == "pre-existing"
    assert judged["2"]["answer"] == 7


def test_judge_skips_queries_without_candidates_in_manifest(tmp_path, monkeypatch):
    root = _make_nopv_dataset(tmp_path)
    monkeypatch.setattr(
        judge, "_extract_pdf_text_for_prompt", lambda pdf_path: "[DOC]\nbody"
    )
    call_count = {"n": 0}

    def fake_call(self, prompt, **kwargs):
        call_count["n"] += 1
        return CacheResult(
            response=json.dumps(
                {
                    "reasoning": "r",
                    "picked_index": 0,
                    "picked_source": "single",
                    "answer": "alpha",
                    "support": "p1",
                }
            ),
            input_tokens=1,
            output_tokens=1,
            latency_ms=0.5,
            cache_hit=False,
        )

    monkeypatch.setattr(generator.AzureResponsesTextCacheCaller, "call", fake_call)

    # Manifest covers only query 1; query 2 should be skipped without LLM call.
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        root,
        candidates={"1": [{"source": "single", "answer": "alpha"}]},
    )

    summary = judge.judge_ground_truth_from_manifest(
        target_dir=root,
        manifest_path=manifest_path,
        query_indices=[1, 2],
        cache_db=str(tmp_path / "judge.db"),
    )

    assert call_count["n"] == 1
    assert summary.judged_count == 1
    assert summary.skipped_no_candidates_count == 1


def test_judge_cache_hit_on_second_invocation(tmp_path, monkeypatch):
    root = _make_nopv_dataset(tmp_path)
    monkeypatch.setattr(
        judge, "_extract_pdf_text_for_prompt", lambda pdf_path: "[DOC]\nstable"
    )
    state = {"n": 0}

    def fake_call(self, prompt, **kwargs):
        state["n"] += 1
        return CacheResult(
            response=json.dumps(
                {
                    "reasoning": "r",
                    "picked_index": 0,
                    "picked_source": "single",
                    "answer": "alpha",
                    "support": "p1",
                }
            ),
            input_tokens=1,
            output_tokens=1,
            latency_ms=0.5,
            cache_hit=(state["n"] > 1),
        )

    monkeypatch.setattr(generator.AzureResponsesTextCacheCaller, "call", fake_call)

    manifest_path = _write_manifest(tmp_path / "manifest.json", root)
    kwargs = dict(
        target_dir=root,
        manifest_path=manifest_path,
        query_indices=[1],
        cache_db=str(tmp_path / "judge.db"),
    )

    first = judge.judge_ground_truth_from_manifest(**kwargs)
    # Erase the output so the second pass actually re-invokes the caller.
    (root / "judged" / "doc_a.judged_answers.json").unlink()
    second = judge.judge_ground_truth_from_manifest(**kwargs)

    assert first.results[0].cache_hit is False
    assert second.results[0].cache_hit is True
    assert state["n"] == 2


def test_judge_log_file_uses_gt_judge_prefix(tmp_path, monkeypatch):
    root = _make_nopv_dataset(tmp_path)
    monkeypatch.setattr(
        judge, "_extract_pdf_text_for_prompt", lambda pdf_path: "[DOC]\nbody"
    )
    monkeypatch.setattr(
        generator.AzureResponsesTextCacheCaller,
        "call",
        lambda self, prompt, **kwargs: CacheResult(
            response=json.dumps(
                {
                    "reasoning": "r",
                    "picked_index": 0,
                    "picked_source": "single",
                    "answer": "alpha",
                    "support": "p1",
                }
            ),
            input_tokens=1,
            output_tokens=1,
            latency_ms=0.5,
            cache_hit=False,
        ),
    )

    manifest_path = _write_manifest(tmp_path / "manifest.json", root)
    summary = judge.judge_ground_truth_from_manifest(
        target_dir=root,
        manifest_path=manifest_path,
        query_indices=[1],
        cache_db=str(tmp_path / "judge.db"),
        log_dir=tmp_path / "logs",
    )

    assert summary.log_path is not None
    assert summary.log_path.name.startswith("gt_judge_")
    assert summary.log_path.exists()
    log_text = summary.log_path.read_text(encoding="utf-8")
    assert "generation_mode=per-query" in log_text


def test_judge_cli_parser_defaults():
    args = judge._build_parser().parse_args(
        [
            "--target-dir",
            "datasets/nopv",
            "--candidates",
            "manifest.json",
            "--query-idx",
            "1",
        ]
    )
    assert args.judge_mode == "per-query"
    assert args.llm_provider == "azure"
    assert args.model == generator.DEFAULT_MODEL
    assert args.input_mode == generator.DEFAULT_INPUT_MODE
