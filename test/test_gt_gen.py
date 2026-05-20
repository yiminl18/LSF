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


def _make_dataset(
    tmp_path: Path,
    names: tuple[str, ...] = ("doc_a",),
    dataset_name: str = "court",
) -> Path:
    root = tmp_path / dataset_name / "latest"
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


def _patch_text_extraction(
    monkeypatch: pytest.MonkeyPatch,
    text: str = "[DOCUMENT TEXT START]\nmock document text",
) -> None:
    monkeypatch.setattr(generator, "_extract_pdf_text_for_prompt", lambda pdf_path: text)


def _patch_azure_text_caller(monkeypatch: pytest.MonkeyPatch, fake_call) -> None:
    def wrapped_call(self, prompt, **kwargs):
        return fake_call(self, prompt=prompt, **kwargs)

    monkeypatch.setattr(generator.AzureResponsesTextCacheCaller, "call", wrapped_call)


def _doc_id_from_prompt(prompt: str) -> str:
    return prompt.split("Document id: ", 1)[1].splitlines()[0]


def test_generate_ground_truth_defaults_to_text_mini(tmp_path, monkeypatch):
    root = _make_dataset(tmp_path)
    seen = {}

    def fake_call(self, **kwargs):
        seen.update(kwargs)
        return CacheResult(
            response='{"reasoning":"cover page","answer":"23-35560"}',
            input_tokens=10,
            cached_input_tokens=6,
            output_tokens=5,
            latency_ms=1.0,
            cache_hit=False,
            cost_usd=0.00003,
        )

    _patch_text_extraction(monkeypatch)
    _patch_azure_text_caller(monkeypatch, fake_call)

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
    assert seen["response_schema"]["required"] == ["reasoning", "answer"]
    assert "support" not in seen["response_schema"]["properties"]
    assert "[DOCUMENT TEXT START]" in seen["prompt"]
    assert "Answer type: string" in seen["prompt"]
    assert '"support"' not in seen["prompt"]
    assert "NOPV labeling rules" not in seen["prompt"]
    assert "One-shot" not in seen["prompt"]


def test_generate_ground_truth_merges_existing_keys(tmp_path, monkeypatch):
    root = _make_dataset(tmp_path)
    gt_dir = root / "ground_truth"
    gt_dir.mkdir()
    (gt_dir / "doc_a.txt_answers.json").write_text(
        json.dumps({"2": "old answer"}), encoding="utf-8"
    )

    _patch_text_extraction(monkeypatch)
    _patch_azure_text_caller(
        monkeypatch,
        lambda self, **kwargs: CacheResult(
            response='{"reasoning":"cover page","answer":["23-35560","23-35585"]}',
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
        calls.append(_doc_id_from_prompt(kwargs["prompt"]))
        return CacheResult(
            response='{"reasoning":"page 1","answer":"new"}',
            input_tokens=10,
            output_tokens=5,
            latency_ms=1.0,
            cache_hit=False,
        )

    _patch_text_extraction(monkeypatch)
    _patch_azure_text_caller(monkeypatch, fake_call)

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
        generator.parse_answer_response('{"reasoning":"page 1"}')


def test_cli_parser_defaults_to_text_mini():
    args = generator._build_parser().parse_args(
        ["--target-dir", "datasets/court", "--query-idx", "1", "--progress-cost"]
    )
    assert args.llm_provider == "azure"
    assert args.model == "gpt-5.4-mini"
    assert args.input_mode == "text"
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
    assert generator.resolve_input_mode("azure", "auto") == "text"
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
                    "result": '{"reasoning":"page 1","answer":"ok"}',
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
                    "result": '{"reasoning":"cover","answer":["23-35560"]}',
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
                    "result": '{"reasoning":"page 1","answer":"cached"}',
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
        calls.append((_doc_id_from_prompt(prompt), query_idx))
        return CacheResult(
            response=f'{{"reasoning":"page 1","answer":"answer-{query_idx}"}}',
            input_tokens=10,
            output_tokens=5,
            latency_ms=1.0,
            cache_hit=False,
        )

    _patch_text_extraction(monkeypatch)
    _patch_azure_text_caller(monkeypatch, fake_call)

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
        calls.append(_doc_id_from_prompt(prompt))
        assert "Question index: 1" in prompt
        assert "Question index: 2" in prompt
        assert "answers" in kwargs["response_schema"]["properties"]
        return CacheResult(
            response=json.dumps(
                {
                    "answers": [
                        {"query_idx": 1, "reasoning": "page 1", "answer": "docket"},
                        {"query_idx": 2, "reasoning": "page 2", "answer": "judge"},
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

    _patch_text_extraction(monkeypatch)
    _patch_azure_text_caller(monkeypatch, fake_call)

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


def test_generation_mode_all_passes_temperature(tmp_path, monkeypatch):
    root = _make_dataset(tmp_path)
    seen_temperatures = []

    def fake_call(self, **kwargs):
        seen_temperatures.append(kwargs["temperature"])
        return CacheResult(
            response=json.dumps(
                {
                    "answers": [
                        {"query_idx": 1, "reasoning": "page 1", "answer": "docket"},
                        {"query_idx": 2, "reasoning": "page 2", "answer": "judge"},
                    ]
                }
            ),
            input_tokens=100,
            output_tokens=20,
            latency_ms=1.0,
            cache_hit=False,
        )

    _patch_text_extraction(monkeypatch)
    _patch_azure_text_caller(monkeypatch, fake_call)

    summary = generator.generate_ground_truth_for_queries(
        target_dir=root,
        query_indices=[1, 2],
        generation_mode="all",
        temperature=0.2,
        cache_db=str(tmp_path / "cache.db"),
    )

    assert summary.generated_count == 2
    assert seen_temperatures == [0.2]


def test_generate_ground_truth_writes_latency_cost_log(tmp_path, monkeypatch):
    root = _make_dataset(tmp_path)

    def fake_call(self, **kwargs):
        return CacheResult(
            response='{"reasoning":"cover page","answer":"23-35560"}',
            input_tokens=10,
            cached_input_tokens=6,
            output_tokens=5,
            latency_ms=1.0,
            cache_hit=False,
            cost_usd=0.00003,
        )

    _patch_text_extraction(monkeypatch)
    _patch_azure_text_caller(monkeypatch, fake_call)

    summary = generator.generate_ground_truth(
        target_dir=root,
        query_idx=1,
        num_doc=1,
        cache_db=str(tmp_path / "cache.db"),
        log_dir=tmp_path / "logs",
    )

    assert summary.log_path is not None
    log_text = summary.log_path.read_text(encoding="utf-8")
    assert "event=run_start" in log_text
    assert "llm_provider=azure" in log_text
    assert "model=gpt-5.4-mini" in log_text
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

    _patch_text_extraction(monkeypatch)
    _patch_azure_text_caller(monkeypatch, fake_call)

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
                        {"query_idx": 1, "reasoning": "page 1", "answer": "docket"},
                        {"query_idx": 2, "reasoning": "page 2", "answer": "judge"},
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

    _patch_text_extraction(monkeypatch)
    _patch_azure_text_caller(monkeypatch, fake_call)

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
    assert "llm_provider=azure" in log_text
    assert "model=gpt-5.4-mini" in log_text
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
                        {"query_idx": 2, "reasoning": "page 2", "answer": "new"},
                    ]
                }
            ),
            input_tokens=100,
            output_tokens=10,
            latency_ms=1.0,
            cache_hit=False,
        )

    _patch_text_extraction(monkeypatch)
    _patch_azure_text_caller(monkeypatch, fake_call)

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
            '{"answers":[{"query_idx":1,"reasoning":"s","answer":"a"}]}',
            queries,
        )
    with pytest.raises(ValueError, match="Duplicate answer"):
        generator.parse_answers_response(
            '{"answers":[{"query_idx":1,"reasoning":"s","answer":"a"},'
            '{"query_idx":1,"reasoning":"s","answer":"b"}]}',
            [queries[0]],
        )
    with pytest.raises(ValueError, match="Unexpected answer"):
        generator.parse_answers_response(
            '{"answers":[{"query_idx":3,"reasoning":"s","answer":"a"}]}',
            [queries[0]],
        )


def test_structured_output_answer_fields_have_types():
    single_schema = generator._answer_response_schema()
    multi_schema = generator._answers_response_schema(
        [generator.QuerySpec(idx=1, text="Q1", answer_type="string")]
    )

    assert "type" in single_schema["properties"]["answer"]
    assert single_schema["properties"]["reasoning"]["type"] == "string"
    assert "support" not in single_schema["properties"]
    assert single_schema["required"] == ["reasoning", "answer"]
    assert (
        "type"
        in multi_schema["properties"]["answers"]["items"]["properties"]["answer"]
    )
    multi_item_schema = multi_schema["properties"]["answers"]["items"]
    assert multi_item_schema["properties"]["reasoning"]["type"] == "string"
    assert "support" not in multi_item_schema["properties"]
    assert multi_item_schema["required"] == [
        "query_idx",
        "reasoning",
        "answer",
    ]


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


def test_nopv_text_prompt_puts_document_before_one_shot_for_prefix_cache():
    query = generator.QuerySpec(
        idx=11,
        text="What mandatory written submissions are required?",
        answer_type="list of strings",
    )
    nopv_template = generator._load_gt_prompt_template("nopv")
    nopv_examples = generator._load_gt_examples("nopv")

    single_prompt = generator.build_ground_truth_text_prompt(
        query=query,
        doc_id="some_other_doc",
        document_text="[DOCUMENT TEXT START]\nshared document text",
        prompt_template=nopv_template,
        examples=nopv_examples,
    )
    all_prompt = generator.build_ground_truth_all_text_prompt(
        queries=[query],
        doc_id="some_other_doc",
        document_text="[DOCUMENT TEXT START]\nshared document text",
        prompt_template=nopv_template,
        examples=nopv_examples,
    )

    assert single_prompt.index("shared document text") < single_prompt.index("One-shot")
    assert single_prompt.index("One-shot") < single_prompt.index("[QUESTION]")
    assert all_prompt.index("shared document text") < all_prompt.index("One-shot")
    assert all_prompt.index("One-shot") < all_prompt.index("[QUESTIONS]")


def test_dataset_prompt_template_is_selected_by_dataset_name():
    query = generator.QuerySpec(
        idx=11,
        text="What mandatory written submissions are required?",
        answer_type="list of strings",
    )
    nopv_template = generator._load_gt_prompt_template("nopv")
    nopv_examples = generator._load_gt_examples("nopv")
    court_template = generator._load_gt_prompt_template("court")

    nopv_prompt = generator.build_ground_truth_prompt(
        query=query,
        doc_id="doc_a",
        prompt_template=nopv_template,
        examples=nopv_examples,
    )
    court_prompt = generator.build_ground_truth_prompt(
        query=query,
        doc_id="doc_a",
        prompt_template=court_template,
    )
    nopv_all_prompt = generator.build_ground_truth_all_prompt(
        queries=[query],
        doc_id="doc_a",
        prompt_template=nopv_template,
        examples=nopv_examples,
    )
    court_all_prompt = generator.build_ground_truth_all_prompt(
        queries=[query],
        doc_id="doc_a",
        prompt_template=court_template,
    )

    assert "NOPV labeling rules:" in nopv_prompt
    assert "not mandated" in nopv_prompt
    assert "NOPV labeling rules:" in nopv_all_prompt
    assert "NOPV labeling rules" not in court_prompt
    assert "NOPV labeling rules" not in court_all_prompt
    assert "not mandated" not in court_prompt
    assert "not mandated" not in court_all_prompt
    assert "One-shot" in nopv_prompt
    assert "12025006NOPV_PCO_05082025" in nopv_prompt
    assert "updated training program" in nopv_prompt
    assert "End of example." in nopv_prompt
    assert "One-shot" in nopv_all_prompt
    assert "Dresser Style 63" in nopv_all_prompt
    for expected_idx in range(1, 14):
        assert f"Question index: {expected_idx}" in nopv_all_prompt
    assert "One-shot" not in court_prompt
    assert "One-shot" not in court_all_prompt
    assert "{{one_shot_example_block}}" not in nopv_prompt
    assert "{{one_shot_example_block}}" not in nopv_all_prompt
    assert nopv_prompt.index('"reasoning"') < nopv_prompt.index('"answer"')
    assert nopv_all_prompt.index('"reasoning"') < nopv_all_prompt.index('"answer"')
    assert '"support"' not in nopv_prompt
    assert '"support"' not in nopv_all_prompt


def test_nopv_one_shot_example_skipped_when_idx_missing():
    query = generator.QuerySpec(
        idx=999,
        text="Hypothetical query not present in the example bank.",
        answer_type="string",
    )
    nopv_template = generator._load_gt_prompt_template("nopv")
    nopv_examples = generator._load_gt_examples("nopv")

    prompt = generator.build_ground_truth_prompt(
        query=query,
        doc_id="some_other_doc",
        prompt_template=nopv_template,
        examples=nopv_examples,
    )

    assert "One-shot" not in prompt
    assert "{{one_shot_example_block}}" not in prompt
    assert "Question index: 999" in prompt
    assert "Document id: some_other_doc" in prompt


def test_nopv_one_shot_example_skipped_when_doc_id_matches():
    nopv_examples = generator._load_gt_examples("nopv")
    assert nopv_examples is not None
    same_doc_id = nopv_examples["doc_id"]
    query = generator.QuerySpec(idx=8, text="?", answer_type="integer")
    nopv_template = generator._load_gt_prompt_template("nopv")

    prompt = generator.build_ground_truth_prompt(
        query=query,
        doc_id=same_doc_id,
        prompt_template=nopv_template,
        examples=nopv_examples,
    )

    assert "One-shot" not in prompt
    assert f"Document id: {same_doc_id}" in prompt


def test_nopv_all_example_lists_designated_query_ids():
    nopv_template = generator._load_gt_prompt_template("nopv")
    nopv_examples = generator._load_gt_examples("nopv")
    live_queries = [
        generator.QuerySpec(idx=1, text="Live Q1", answer_type="string"),
        generator.QuerySpec(idx=2, text="Live Q2", answer_type="string"),
    ]

    prompt = generator.build_ground_truth_all_prompt(
        queries=live_queries,
        doc_id="some_other_doc",
        prompt_template=nopv_template,
        examples=nopv_examples,
    )

    assert "One-shot" in prompt
    one_shot_start = prompt.index("One-shot")
    one_shot_end = prompt.index("End of example.", one_shot_start)
    one_shot_block = prompt[one_shot_start:one_shot_end]
    assert nopv_examples["all_example_query_ids"] == list(range(1, 14))
    for expected_idx in range(1, 14):
        assert f"Question index: {expected_idx}" in one_shot_block
        assert f'"query_idx": {expected_idx}' in one_shot_block
    assert "§ 192.605" in one_shot_block
    assert "Dresser Style 63" in one_shot_block
    assert "updated training program" in one_shot_block
    # Live queries appear only after the example block.
    assert "Live Q1" not in one_shot_block
    assert "Question index: 1" in prompt[one_shot_end:]


def test_generate_ground_truth_loads_source_prompt_template(tmp_path, monkeypatch):
    root = _make_dataset(tmp_path, dataset_name="nopv")
    seen = {}

    def fake_call(self, **kwargs):
        seen["prompt"] = kwargs["prompt"]
        return CacheResult(
            response='{"reasoning":"page 1","answer":"ok"}',
            input_tokens=10,
            output_tokens=5,
            latency_ms=1.0,
            cache_hit=False,
        )

    _patch_text_extraction(monkeypatch)
    _patch_azure_text_caller(monkeypatch, fake_call)

    summary = generator.generate_ground_truth(
        target_dir=root.parent,
        query_idx=1,
        num_doc=1,
        cache_db=str(tmp_path / "cache.db"),
    )

    assert summary.generated_count == 1
    assert "NOPV labeling rules" in seen["prompt"]
    assert "safety-improvement-cost" in seen["prompt"]
    assert "One-shot" in seen["prompt"]
    assert "May 8, 2025" in seen["prompt"]
    assert "{{one_shot_example_block}}" not in seen["prompt"]
    assert seen["prompt"].index('"reasoning"') < seen["prompt"].index('"answer"')
    assert '"support"' not in seen["prompt"]


def test_build_azure_call_result_uses_cached_token_usage():
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=200,
        input_tokens_details=SimpleNamespace(cached_tokens=600),
    )

    result = generator._build_azure_call_result(
        answer='{"reasoning":"page 1","answer":"ok"}',
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
            response='{"reasoning":"page 1","answer":"ok"}',
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
                output_text='{"reasoning":"page 1","answer":"ok"}',
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
    assert seen["text"]["format"]["schema"]["required"] == [
        "reasoning",
        "answer",
    ]
    assert "support" not in seen["text"]["format"]["schema"]["properties"]
