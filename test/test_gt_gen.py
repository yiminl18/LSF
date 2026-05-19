from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from core.pipeline.e2e_utils.cache import CacheResult
from gt_gen import generator


@pytest.fixture(autouse=True)
def _isolate_default_log_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _make_dataset(tmp_path: Path, names: tuple[str, ...] = ("doc_a",)) -> Path:
    root = tmp_path / "court" / "latest"
    raw = root / "raw"
    raw.mkdir(parents=True)
    (root / "queries.json").write_text(
        json.dumps(
            [
                {"text": "What is the docket number?", "answer_type": "string"},
                {"text": "Who authored the opinion?", "answer_type": "string"},
            ]
        ),
        encoding="utf-8",
    )
    for name in names:
        (raw / f"{name}.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    return root


def test_generate_ground_truth_defaults_to_native_pdf_mini(tmp_path, monkeypatch):
    root = _make_dataset(tmp_path)
    seen = {}

    def fake_call(self, **kwargs):
        seen.update(kwargs)
        return CacheResult(
            response='{"answer":"23-35560","support":"cover page"}',
            input_tokens=10,
            cached_input_tokens=6,
            output_tokens=5,
            latency_ms=1.0,
            cache_hit=False,
            cost_usd=0.00003,
        )

    monkeypatch.setattr(generator.NativePDFCacheCaller, "call", fake_call)

    summary = generator.generate_ground_truth(
        target_dir=root.parent,
        query_idx=1,
        num_doc=1,
        cache_db=str(tmp_path / "cache.db"),
    )

    output = json.loads(
        (root / "ground_truth" / "doc_a.txt_answers.json").read_text(
            encoding="utf-8"
        )
    )
    assert output == {"1": "23-35560"}
    assert summary.generated_count == 1
    assert summary.results[0].cached_input_tokens == 6
    assert summary.results[0].cost_usd == pytest.approx(0.00003)
    assert summary.log_path is not None
    assert summary.log_path.parent == Path(generator.DEFAULT_LOG_DIR)
    assert summary.log_path.exists()
    assert seen["llm_provider"] == "azure"
    assert seen["model"] == "gpt-5.4-mini"
    assert seen["pdf_path"] == root / "raw" / "doc_a.pdf"
    assert seen["response_schema"]["required"] == ["answer", "support"]
    assert "Answer type: string" in seen["prompt"]


def test_generate_ground_truth_merges_existing_keys(tmp_path, monkeypatch):
    root = _make_dataset(tmp_path)
    gt_dir = root / "ground_truth"
    gt_dir.mkdir()
    (gt_dir / "doc_a.txt_answers.json").write_text(
        json.dumps({"2": "old answer"}), encoding="utf-8"
    )

    monkeypatch.setattr(
        generator.NativePDFCacheCaller,
        "call",
        lambda self, **kwargs: CacheResult(
            response='{"answer":["23-35560","23-35585"],"support":"cover page"}',
            input_tokens=10,
            output_tokens=5,
            latency_ms=1.0,
            cache_hit=True,
        ),
    )

    summary = generator.generate_ground_truth(
        target_dir=root,
        query_idx=1,
        cache_db=str(tmp_path / "cache.db"),
    )

    output = json.loads((gt_dir / "doc_a.txt_answers.json").read_text(encoding="utf-8"))
    assert output == {"1": ["23-35560", "23-35585"], "2": "old answer"}
    assert summary.generated_count == 1
    assert summary.results[0].cache_hit is True


def test_generate_ground_truth_samples_then_skips_existing(tmp_path, monkeypatch):
    root = _make_dataset(tmp_path, names=("doc_a", "doc_b"))
    gt_dir = root / "ground_truth"
    gt_dir.mkdir()
    (gt_dir / "doc_a.txt_answers.json").write_text(
        json.dumps({"1": "existing"}), encoding="utf-8"
    )
    calls = []

    def fake_call(self, **kwargs):
        calls.append(kwargs["pdf_path"].stem)
        return CacheResult(
            response='{"answer":"new","support":"page 1"}',
            input_tokens=10,
            output_tokens=5,
            latency_ms=1.0,
            cache_hit=False,
        )

    monkeypatch.setattr(generator.NativePDFCacheCaller, "call", fake_call)

    summary = generator.generate_ground_truth(
        target_dir=root,
        query_idx=1,
        num_doc=2,
        cache_db=str(tmp_path / "cache.db"),
    )

    assert calls == ["doc_b"]
    assert summary.selected_count == 2
    assert summary.skipped_existing_count == 1
    assert summary.generated_count == 1
    assert json.loads((gt_dir / "doc_a.txt_answers.json").read_text()) == {
        "1": "existing"
    }


def test_parse_answer_response_requires_answer_field():
    with pytest.raises(ValueError, match="answer"):
        generator.parse_answer_response('{"support":"page 1"}')


def test_cli_parser_defaults_to_native_pdf_mini():
    args = generator._build_parser().parse_args(
        ["--target-dir", "datasets/court", "--query-idx", "1", "--progress-cost"]
    )
    assert args.llm_provider == "azure"
    assert args.model == "gpt-5.4-mini"
    assert args.input_mode == "auto"
    assert args.generation_mode == "single"
    assert args.progress_cost is True
    assert args.log_dir == generator.DEFAULT_LOG_DIR


def test_cli_parser_accepts_query_indices():
    args = generator._build_parser().parse_args(
        [
            "--target-dir",
            "datasets/court",
            "--query-indices",
            "1-2",
            "--generation-mode",
            "all",
        ]
    )
    assert args.query_idx is None
    assert args.query_indices == "1-2"
    assert args.generation_mode == "all"
    assert generator._parse_query_indices_arg("1,3-4") == [1, 3, 4]
    assert generator._parse_query_indices_arg("all") is None


def test_provider_auto_input_mode_resolution():
    assert generator.resolve_input_mode("azure", "auto") == "native-pdf"
    assert generator.resolve_input_mode("claude-code", "auto") == "text"
    assert generator.resolve_model_for_provider("claude-code", "gpt-5.4-mini") == "sonnet"


def test_claude_code_text_mode_passes_prompt_via_stdin(tmp_path, monkeypatch):
    root = _make_dataset(tmp_path)
    seen = {}

    monkeypatch.setattr(
        generator,
        "_extract_pdf_text_for_prompt",
        lambda pdf_path: "[DOCUMENT TEXT START]\nsecret document text",
    )

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["input"] = kwargs["input"]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "is_error": False,
                    "result": '{"answer":"ok","support":"page 1"}',
                    "usage": {
                        "input_tokens": 2,
                        "cache_creation_input_tokens": 3,
                        "cache_read_input_tokens": 4,
                        "output_tokens": 5,
                    },
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(generator.subprocess, "run", fake_run)

    summary = generator.generate_ground_truth(
        target_dir=root,
        query_idx=1,
        num_doc=1,
        llm_provider="claude-code",
        cache_db=str(tmp_path / "cache.db"),
    )

    assert summary.generated_count == 1
    assert seen["cmd"][:4] == ["claude", "-p", "--model", "sonnet"]
    assert "secret document text" in seen["input"]
    assert "secret document text" not in " ".join(seen["cmd"])
    assert seen["input"].index("secret document text") < seen["input"].index(
        "Question:"
    )
    output = json.loads(
        (root / "ground_truth" / "doc_a.txt_answers.json").read_text(
            encoding="utf-8"
        )
    )
    assert output == {"1": "ok"}


def test_claude_code_read_pdf_mode_allows_read_tool(tmp_path, monkeypatch):
    root = _make_dataset(tmp_path)
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["input"] = kwargs["input"]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "is_error": False,
                    "result": '{"answer":["23-35560"],"support":"cover"}',
                    "usage": {"input_tokens": 10, "output_tokens": 3},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(generator.subprocess, "run", fake_run)

    summary = generator.generate_ground_truth(
        target_dir=root,
        query_idx=1,
        num_doc=1,
        llm_provider="claude-code",
        model="opus",
        input_mode="claude-read-pdf",
        cache_db=str(tmp_path / "cache.db"),
    )

    assert summary.generated_count == 1
    assert seen["cmd"][:4] == ["claude", "-p", "--model", "opus"]
    assert "--tools=Read" in seen["cmd"]
    assert "--add-dir" in seen["cmd"]
    assert str(root / "raw" / "doc_a.pdf") in seen["input"]


def test_claude_code_cache_hit_skips_subprocess(tmp_path, monkeypatch):
    root = _make_dataset(tmp_path)
    calls = {"count": 0}
    monkeypatch.setattr(generator, "_extract_pdf_text_for_prompt", lambda pdf_path: "doc")

    def fake_run(cmd, **kwargs):
        calls["count"] += 1
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "is_error": False,
                    "result": '{"answer":"cached","support":"page 1"}',
                    "usage": {"input_tokens": 2, "output_tokens": 1},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(generator.subprocess, "run", fake_run)
    kwargs = {
        "target_dir": root,
        "query_idx": 1,
        "num_doc": 1,
        "llm_provider": "claude-code",
        "cache_db": str(tmp_path / "cache.db"),
    }

    first = generator.generate_ground_truth(**kwargs)
    (root / "ground_truth" / "doc_a.txt_answers.json").unlink()
    second = generator.generate_ground_truth(**kwargs)

    assert first.results[0].cache_hit is False
    assert second.results[0].cache_hit is True
    assert calls["count"] == 1


def test_generate_ground_truth_for_queries_uses_doc_major_order(tmp_path, monkeypatch):
    root = _make_dataset(tmp_path, names=("doc_a", "doc_b"))
    calls = []

    def fake_call(self, **kwargs):
        prompt = kwargs["prompt"]
        query_idx = int(prompt.split("Question index: ", 1)[1].splitlines()[0])
        calls.append((kwargs["pdf_path"].stem, query_idx))
        return CacheResult(
            response=f'{{"answer":"answer-{query_idx}","support":"page 1"}}',
            input_tokens=10,
            output_tokens=5,
            latency_ms=1.0,
            cache_hit=False,
        )

    monkeypatch.setattr(generator.NativePDFCacheCaller, "call", fake_call)

    summary = generator.generate_ground_truth_for_queries(
        target_dir=root,
        query_indices=[1, 2],
        num_doc=2,
        cache_db=str(tmp_path / "cache.db"),
    )

    assert calls == [("doc_a", 1), ("doc_a", 2), ("doc_b", 1), ("doc_b", 2)]
    assert summary.generated_count == 4
    assert json.loads(
        (root / "ground_truth" / "doc_a.txt_answers.json").read_text(
            encoding="utf-8"
        )
    ) == {"1": "answer-1", "2": "answer-2"}


def test_generation_mode_all_answers_selected_queries_once_per_doc(tmp_path, monkeypatch):
    root = _make_dataset(tmp_path, names=("doc_a", "doc_b"))
    calls = []

    def fake_call(self, **kwargs):
        prompt = kwargs["prompt"]
        calls.append(kwargs["pdf_path"].stem)
        assert "Question index: 1" in prompt
        assert "Question index: 2" in prompt
        assert "answers" in kwargs["response_schema"]["properties"]
        return CacheResult(
            response=json.dumps(
                {
                    "answers": [
                        {"query_idx": 1, "answer": "docket", "support": "page 1"},
                        {"query_idx": 2, "answer": "judge", "support": "page 2"},
                    ]
                }
            ),
            input_tokens=100,
            cached_input_tokens=50,
            output_tokens=20,
            latency_ms=1.0,
            cache_hit=False,
            cost_usd=0.01,
        )

    monkeypatch.setattr(generator.NativePDFCacheCaller, "call", fake_call)

    summary = generator.generate_ground_truth_for_queries(
        target_dir=root,
        query_indices=[1, 2],
        num_doc=2,
        generation_mode="all",
        cache_db=str(tmp_path / "cache.db"),
    )

    assert calls == ["doc_a", "doc_b"]
    assert summary.generated_count == 4
    assert json.loads(
        (root / "ground_truth" / "doc_a.txt_answers.json").read_text(
            encoding="utf-8"
        )
    ) == {"1": "docket", "2": "judge"}
    assert generator._api_usage_totals(summary.results)[:4] == (200, 100, 40, 0.02)


def test_generate_ground_truth_writes_latency_cost_log(tmp_path, monkeypatch):
    root = _make_dataset(tmp_path)

    def fake_call(self, **kwargs):
        return CacheResult(
            response='{"answer":"23-35560","support":"cover page"}',
            input_tokens=10,
            cached_input_tokens=6,
            output_tokens=5,
            latency_ms=1.0,
            cache_hit=False,
            cost_usd=0.00003,
        )

    monkeypatch.setattr(generator.NativePDFCacheCaller, "call", fake_call)

    summary = generator.generate_ground_truth(
        target_dir=root,
        query_idx=1,
        num_doc=1,
        cache_db=str(tmp_path / "cache.db"),
        log_dir=tmp_path / "logs",
    )

    assert summary.log_path is not None
    log_text = summary.log_path.read_text(encoding="utf-8")
    assert "event=llm_call" in log_text
    assert "latency_ms=1.000" in log_text
    assert "cost_usd=0.000030" in log_text
    assert "event=run_summary" in log_text
    assert "api_latency_ms=1.000" in log_text
    assert "run_latency_ms=" in log_text


def test_post_call_parse_failure_keeps_latency_cost_in_log(tmp_path, monkeypatch):
    root = _make_dataset(tmp_path)

    def fake_call(self, **kwargs):
        return CacheResult(
            response="not json",
            input_tokens=10,
            cached_input_tokens=6,
            output_tokens=5,
            latency_ms=1.0,
            cache_hit=False,
            cost_usd=0.00003,
        )

    monkeypatch.setattr(generator.NativePDFCacheCaller, "call", fake_call)

    summary = generator.generate_ground_truth(
        target_dir=root,
        query_idx=1,
        num_doc=1,
        cache_db=str(tmp_path / "cache.db"),
        log_dir=tmp_path / "logs",
    )

    assert summary.failed_count == 1
    result = summary.results[0]
    assert result.status == "failed"
    assert result.input_tokens == 10
    assert result.cached_input_tokens == 6
    assert result.output_tokens == 5
    assert result.latency_ms == pytest.approx(1.0)
    assert result.cost_usd == pytest.approx(0.00003)
    assert generator._api_usage_totals(summary.results)[:4] == (
        10,
        6,
        5,
        0.00003,
    )

    assert summary.log_path is not None
    log_text = summary.log_path.read_text(encoding="utf-8")
    assert "event=llm_call" in log_text
    assert "status=failed" in log_text
    assert "latency_ms=1.000" in log_text
    assert "cost_usd=0.000030" in log_text
    assert "api_latency_ms=1.000" in log_text


def test_generation_mode_all_logs_one_llm_call_per_doc(tmp_path, monkeypatch):
    root = _make_dataset(tmp_path)

    def fake_call(self, **kwargs):
        return CacheResult(
            response=json.dumps(
                {
                    "answers": [
                        {"query_idx": 1, "answer": "docket", "support": "page 1"},
                        {"query_idx": 2, "answer": "judge", "support": "page 2"},
                    ]
                }
            ),
            input_tokens=100,
            cached_input_tokens=50,
            output_tokens=20,
            latency_ms=7.5,
            cache_hit=False,
            cost_usd=0.01,
        )

    monkeypatch.setattr(generator.NativePDFCacheCaller, "call", fake_call)

    summary = generator.generate_ground_truth_for_queries(
        target_dir=root,
        query_indices=[1, 2],
        generation_mode="all",
        cache_db=str(tmp_path / "cache.db"),
        log_dir=tmp_path / "logs",
    )

    assert summary.log_path is not None
    log_text = summary.log_path.read_text(encoding="utf-8")
    assert log_text.count("event=llm_call") == 1
    assert "query_ids=1,2" in log_text
    assert "api_latency_ms=7.500" in log_text
    assert "cost_usd=0.010000" in log_text


def test_print_summary_includes_final_latency_cost_and_log_path(tmp_path, capsys):
    summary = generator.GenerationSummary(
        dataset_root=tmp_path / "court" / "latest",
        query=generator.QuerySpec(idx=1, text="Q1", answer_type="string"),
        selected_count=1,
        generated_count=1,
        skipped_existing_count=0,
        failed_count=0,
        run_latency_ms=12.34,
        log_path=tmp_path / "logs" / "run.log",
        results=(
            generator.DocRunResult(
                doc_id="doc_a",
                output_path=tmp_path / "doc_a.txt_answers.json",
                status="generated",
                query_idx=1,
                input_tokens=10,
                cached_input_tokens=6,
                output_tokens=5,
                cost_usd=0.00003,
                latency_ms=1.0,
            ),
        ),
    )

    generator._print_summary(summary)
    out = capsys.readouterr().out

    assert "Run Latency:  12.340 ms" in out
    assert "API Latency:  1.000 ms" in out
    assert "API Cost:     $0.000030" in out
    assert f"Log File:     {summary.log_path}" in out


def test_generation_mode_all_skips_existing_queries(tmp_path, monkeypatch):
    root = _make_dataset(tmp_path)
    gt_dir = root / "ground_truth"
    gt_dir.mkdir()
    (gt_dir / "doc_a.txt_answers.json").write_text(
        json.dumps({"1": "existing"}), encoding="utf-8"
    )
    prompts = []

    def fake_call(self, **kwargs):
        prompts.append(kwargs["prompt"])
        return CacheResult(
            response=json.dumps(
                {
                    "answers": [
                        {"query_idx": 2, "answer": "new", "support": "page 2"},
                    ]
                }
            ),
            input_tokens=100,
            output_tokens=10,
            latency_ms=1.0,
            cache_hit=False,
        )

    monkeypatch.setattr(generator.NativePDFCacheCaller, "call", fake_call)

    summary = generator.generate_ground_truth_for_queries(
        target_dir=root,
        query_indices=[1, 2],
        generation_mode="all",
        cache_db=str(tmp_path / "cache.db"),
    )

    assert len(prompts) == 1
    assert "Question index: 1" not in prompts[0]
    assert "Question index: 2" in prompts[0]
    assert summary.skipped_existing_count == 1
    assert summary.generated_count == 1
    assert json.loads((gt_dir / "doc_a.txt_answers.json").read_text()) == {
        "1": "existing",
        "2": "new",
    }


def test_parse_answers_response_rejects_missing_duplicate_and_unexpected_queries():
    queries = [
        generator.QuerySpec(idx=1, text="Q1", answer_type="string"),
        generator.QuerySpec(idx=2, text="Q2", answer_type="string"),
    ]

    with pytest.raises(ValueError, match="Missing answers"):
        generator.parse_answers_response(
            '{"answers":[{"query_idx":1,"answer":"a","support":"s"}]}',
            queries,
        )
    with pytest.raises(ValueError, match="Duplicate answer"):
        generator.parse_answers_response(
            '{"answers":[{"query_idx":1,"answer":"a","support":"s"},'
            '{"query_idx":1,"answer":"b","support":"s"}]}',
            [queries[0]],
        )
    with pytest.raises(ValueError, match="Unexpected answer"):
        generator.parse_answers_response(
            '{"answers":[{"query_idx":3,"answer":"a","support":"s"}]}',
            [queries[0]],
        )


def test_text_prompt_puts_document_before_query_for_prefix_cache():
    query = generator.QuerySpec(
        idx=2,
        text="Who authored the opinion?",
        answer_type="string",
    )
    prompt = generator.build_ground_truth_text_prompt(
        query=query,
        doc_id="doc_a",
        document_text="[DOCUMENT TEXT START]\nshared document text",
    )

    assert prompt.index("shared document text") < prompt.index("[QUESTION]")
    assert "Question index: 2" in prompt


def test_build_azure_call_result_uses_cached_token_usage():
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=200,
        input_tokens_details=SimpleNamespace(cached_tokens=600),
    )

    result = generator._build_azure_call_result(
        answer='{"answer":"ok","support":"page 1"}',
        usage=usage,
        model="gpt-5.4-mini",
    )

    assert result.input_tokens == 1000
    assert result.cached_input_tokens == 600
    assert result.output_tokens == 200
    assert result.cost_usd == pytest.approx(
        (400 * 0.75 + 600 * 0.075 + 200 * 4.5) / 1_000_000
    )


def test_azure_text_mode_records_exact_responses_usage(tmp_path, monkeypatch):
    root = _make_dataset(tmp_path)
    monkeypatch.setattr(
        generator,
        "_extract_pdf_text_for_prompt",
        lambda pdf_path: "[DOCUMENT TEXT START]\ntext",
    )
    monkeypatch.setattr(
        generator,
        "_azure_responses_text_call",
        lambda **kwargs: generator.AzureResponsesCallResult(
            response='{"answer":"ok","support":"page 1"}',
            input_tokens=100,
            cached_input_tokens=80,
            output_tokens=10,
            cost_usd=0.000123,
        ),
    )

    summary = generator.generate_ground_truth(
        target_dir=root,
        query_idx=1,
        num_doc=1,
        input_mode="text",
        cache_db=str(tmp_path / "cache.db"),
    )

    assert summary.generated_count == 1
    assert summary.results[0].input_tokens == 100
    assert summary.results[0].cached_input_tokens == 80
    assert summary.results[0].output_tokens == 10
    assert summary.results[0].cost_usd == pytest.approx(0.000123)


def test_azure_responses_text_call_passes_structured_output(monkeypatch):
    seen = {}

    class FakeResponses:
        def create(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(
                output_text='{"answer":"ok","support":"page 1"}',
                usage=SimpleNamespace(
                    input_tokens=10,
                    output_tokens=2,
                    input_tokens_details=SimpleNamespace(cached_tokens=0),
                ),
            )

    fake_client = SimpleNamespace(responses=FakeResponses())
    monkeypatch.setattr(
        generator,
        "_azure_responses_client_and_deployment",
        lambda model: (fake_client, "deployment"),
    )

    generator._azure_responses_text_call(
        prompt="prompt",
        model="gpt-5.4-mini",
        max_tokens=20,
        response_schema=generator._answer_response_schema(),
        temperature=0,
    )

    assert seen["text"]["format"]["type"] == "json_schema"
    assert seen["text"]["format"]["name"] == "gt_gen_single_answer"
    assert seen["text"]["format"]["schema"]["required"] == ["answer", "support"]
