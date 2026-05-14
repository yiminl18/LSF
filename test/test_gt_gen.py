from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from core.pipeline.e2e_utils.cache import CacheResult
from gt_gen import generator


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
            output_tokens=5,
            latency_ms=1.0,
            cache_hit=False,
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
    assert seen["llm_provider"] == "azure"
    assert seen["model"] == "gpt-5.4-mini"
    assert seen["pdf_path"] == root / "raw" / "doc_a.pdf"
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
        ["--target-dir", "datasets/court", "--query-idx", "1"]
    )
    assert args.llm_provider == "azure"
    assert args.model == "gpt-5.4-mini"
    assert args.input_mode == "auto"


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
