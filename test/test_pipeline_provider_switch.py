from __future__ import annotations

import importlib
import json
import sys
from types import SimpleNamespace

from core.pipeline import build_processing_json, e2e, generate_labels
from core.retrieval.retrieval import find_provenance_node

judge_header_module = importlib.import_module("core.retrieval.judge_header")


def test_build_processing_json_cli_forwards_llm_provider_and_llm_model(monkeypatch):
    seen = {}

    monkeypatch.setattr(
        build_processing_json,
        "reconstruct_documents",
        lambda **kwargs: seen.update(kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_processing_json",
            "--dataset",
            "pdfs",
            "--parser",
            "docling",
            "--llm-provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
        ],
    )

    build_processing_json.main()

    assert seen["llm_provider"] == "openrouter"
    assert seen["llm_model"] == "z-ai/glm-5.1"


def test_generate_labels_cli_forwards_llm_provider_and_llm_model(monkeypatch):
    seen = {}

    monkeypatch.setattr(
        generate_labels, "generate_labels", lambda **kwargs: seen.update(kwargs)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_labels",
            "--dataset",
            "pdfs",
            "--parser",
            "docling",
            "--judge-mode",
            "answer_compare",
            "--llm-provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
        ],
    )

    generate_labels.main()

    assert seen["llm_provider"] == "openrouter"
    assert seen["llm_model"] == "z-ai/glm-5.1"


def test_e2e_cli_accepts_llm_provider_and_llm_model():
    parser = e2e._build_parser()
    args = parser.parse_args(
        [
            "--dataset",
            "pdfs",
            "--parser",
            "docling",
            "--llm-provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
        ]
    )

    assert args.llm_provider == "openrouter"
    assert args.model == "z-ai/glm-5.1"


def test_judge_header_cli_forwards_llm_provider_and_llm_model(monkeypatch, capsys):
    seen = {}

    monkeypatch.setattr(
        judge_header_module,
        "judge_header",
        lambda text, question, answer, mode, llm_provider, llm_model=None: (
            seen.update(
                {
                    "text": text,
                    "question": question,
                    "answer": answer,
                    "mode": mode,
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                }
            )
            or (True, "ok")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "judge_header",
            "--text",
            "header",
            "--question",
            "q",
            "--answer",
            "a",
            "--llm-provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
        ],
    )

    judge_header_module.main()

    assert seen["llm_provider"] == "openrouter"
    assert seen["llm_model"] == "z-ai/glm-5.1"
    assert "matched=True" in capsys.readouterr().out


def test_llm_model_reaches_llm_call_in_judge_header(monkeypatch):
    seen = {}

    monkeypatch.setattr(judge_header_module, "_check_cache", lambda key: None)
    monkeypatch.setattr(
        judge_header_module,
        "llm_call",
        lambda prompt, llm_provider, model=None, max_tokens=0: (
            seen.update(
                {"llm_provider": llm_provider, "model": model, "max_tokens": max_tokens}
            )
            or "True"
        ),
    )

    matched, raw = judge_header_module.judge_header(
        text="ctx",
        question="q",
        ground_truth="a",
        mode="support_judge",
        llm_provider="openrouter",
        llm_model="z-ai/glm-5.1",
    )

    assert matched is True
    assert raw == "True"
    assert seen == {
        "llm_provider": "openrouter",
        "model": "z-ai/glm-5.1",
        "max_tokens": 10,
    }


def test_llm_model_forwards_to_find_provenance_judge(monkeypatch, tmp_path):
    merged_json = tmp_path / "doc_reconstructed.json"
    merged_json.write_text(
        json.dumps(
            {
                "texts": [
                    {
                        "label": "section_header",
                        "text": "Header",
                        "text_span": "Relevant answer",
                        "structure": {"path_text": "Header"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    seen = {}

    monkeypatch.setattr(
        "core.retrieval.retrieval.load_document_embeddings",
        lambda merged_json_path, cache_dir, provider: (
            {"Header Relevant answer": [1.0]},
            None,
        ),
    )
    monkeypatch.setattr(
        "core.retrieval.retrieval.get_query_embedding",
        lambda question, model, provider: [1.0],
    )
    monkeypatch.setattr(
        "core.retrieval.retrieval.get_model_name_for_provider", lambda provider: "dummy"
    )
    monkeypatch.setattr("core.retrieval.retrieval.cosine_sim", lambda a, b: 0.9)
    monkeypatch.setattr(
        "core.retrieval.retrieval.judge_header",
        lambda text, question, answer, mode, llm_provider, llm_model=None, path_text="": (
            seen.update(
                {
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                    "path_text": path_text,
                }
            )
            or (True, "match")
        ),
    )

    matches, checked = find_provenance_node(
        pdf_path="doc.pdf",
        merged_json_path=str(merged_json),
        question="q",
        answer="a",
        llm_provider="openrouter",
        llm_model="z-ai/glm-5.1",
        top_k_check=1,
        match_limit=1,
    )

    assert len(matches) == 1
    assert checked == 1
    assert seen["llm_provider"] == "openrouter"
    assert seen["llm_model"] == "z-ai/glm-5.1"


def test_e2e_llm_model_propagates_to_generation_and_judge(tmp_path, monkeypatch):
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    (gt_dir / "doc1.txt_answers.json").write_text(
        json.dumps({"1": "answer"}), encoding="utf-8"
    )

    seen = {"gen": None, "judge": None}

    monkeypatch.setattr(
        e2e,
        "generate_answer",
        lambda **kwargs: (
            seen.__setitem__("gen", kwargs["llm_model"])
            or SimpleNamespace(
                answer="answer",
                cost_usd=0.1,
                cache_hit=False,
                latency_ms=1.0,
            )
        ),
    )
    monkeypatch.setattr(
        e2e,
        "judge_answer",
        lambda **kwargs: (
            seen.__setitem__("judge", kwargs["llm_model"])
            or SimpleNamespace(
                is_correct=True,
                raw_response="True",
                latency_ms=1.0,
                cost_usd=0.1,
                cache_hit=False,
            )
        ),
    )

    class _Tracker:
        def record_gen(self, cost: float, cache_hit: bool) -> None:
            return None

        def record_judge(self, cost: float, cache_hit: bool, is_correct: bool) -> None:
            return None

        def record_doc_done(self) -> None:
            return None

        def progress_line(self, q_idx: int, doc_id: str) -> str:
            return f"q{q_idx}:{doc_id}"

    result = e2e._process_doc(
        doc_detail={
            "doc_id": "doc1",
            "methods": {
                "ranker": {
                    "total_candidates": 1,
                    "top_5": [
                        {"path_text": "Header", "text": "Header", "text_span": "Body"}
                    ],
                }
            },
        },
        q_idx=0,
        question="q",
        eval_method="ranker",
        top_k_min=1,
        cached_caller=SimpleNamespace(),
        llm_provider="openrouter",
        llm_model="z-ai/glm-5.1",
        gen_max_tokens=10,
        judge_max_tokens=10,
        gt_dir=gt_dir,
        tracker=_Tracker(),
    )

    assert result is not None
    assert seen == {"gen": "z-ai/glm-5.1", "judge": "z-ai/glm-5.1"}
