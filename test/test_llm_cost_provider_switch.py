from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.llm.cost import compute_cost, get_prices
from core.pipeline import generate_labels
from core.pipeline.e2e_utils.baselines.base import BaseRAGBaseline
from core.pipeline.e2e_utils.cache import CacheResult
from core.pipeline.e2e_utils.generation import generate_answer
from core.pipeline.e2e_utils.judge import judge_answer
from core.pipeline.e2e_utils.rag_vanilla import generate_answer_from_text


class _FakeCachedCaller:
    def __init__(self, response: str, input_tokens: int, output_tokens: int):
        self._result = CacheResult(
            response=response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=12.5,
            cache_hit=False,
        )

    def call(self, *args, **kwargs):
        return self._result


class _StubBaseline(BaseRAGBaseline):
    name = "stub"

    def preprocess_doc(self, doc_id: str) -> None:
        raise NotImplementedError

    def retrieve(
        self, query: str, query_embedding: list[float], doc_id: str, top_k: int
    ):
        raise NotImplementedError


def test_openrouter_glm_pricing():
    assert get_prices("openrouter", "z-ai/glm-5.1") == (0.95, 3.15)


def test_openrouter_glm_pricing_normalizes_prefix_and_case():
    assert get_prices("openrouter", "Z-AI/GLM-5.1") == (0.95, 3.15)
    assert get_prices("openrouter", "openrouter:z-ai/glm-5.1") == (0.95, 3.15)


def test_gpt54_pricing_distinguishes_from_generic_gpt5_series():
    assert get_prices("azure", "gpt-5.4") == (2.5, 15.0)
    assert get_prices("azure", "gpt-5.4-2026-04-14") == (2.5, 15.0)
    assert get_prices("openrouter", "openai/gpt-5.4") == (2.5, 15.0)


def test_gpt54mini_pricing_overrides_generic_mini_pricing():
    assert get_prices("azure", "gpt-5.4-mini") == (0.75, 4.5)
    assert get_prices("azure", "gpt-5.4-mini-2026-04-14") == (0.75, 4.5)
    assert get_prices("openrouter", "openrouter:openai/gpt-5.4-mini") == (0.75, 4.5)


def test_gpt54_prefix_matching_requires_model_boundary():
    assert get_prices("azure", "gpt-5.40") == (2.5, 10.0)
    assert get_prices("azure", "gpt-5.4o-mini") == (0.15, 0.60)


def test_glm_cost_alignment(monkeypatch, tmp_path):
    processing_path = tmp_path / "processing" / "doc1_reconstructed.json"
    processing_path.parent.mkdir(parents=True, exist_ok=True)
    processing_path.write_text(
        json.dumps(
            {
                "texts": [
                    {
                        "label": "section_header",
                        "text": "Header",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakePathManager:
        def __init__(self, experiment: str, processing_variant: str | None):
            self.experiment = experiment
            self.processing_variant = processing_variant

        def get_data_dir(self, dataset: str):
            return tmp_path / "data"

        def get_ground_truth_dir(self, dataset: str):
            return tmp_path / "gt"

        def get_questions_path(self, dataset: str):
            return tmp_path / "questions.txt"

        def get_labels_dir(self, dataset: str):
            return tmp_path / "labels"

        def get_processing_json_path(self, dataset: str, doc_name: str):
            return processing_path

    monkeypatch.setattr(generate_labels, "PathManager", FakePathManager)
    monkeypatch.setattr(
        generate_labels,
        "load_questions",
        lambda path: ["question 1"],
    )
    monkeypatch.setattr(
        generate_labels,
        "match_pdf_to_ground_truth",
        lambda pdf_dir, gt_dir: [(Path("doc1.pdf"), "doc1", ["answer"])],
    )
    monkeypatch.setattr(
        generate_labels,
        "load_existing_labels",
        lambda path, force_without_absence=False: ([], set()),
    )

    seen = {}

    def fake_get_prices(provider: str, model: str = ""):
        seen["provider"] = provider
        seen["model"] = model
        if provider == "openrouter" and model == "z-ai/glm-5.1":
            return 1.23, 4.56
        return 9.87, 6.54

    monkeypatch.setattr(generate_labels, "get_prices", fake_get_prices)

    cost, calls = generate_labels._estimate_cost(
        dataset="pdfs",
        q_idxs_to_run=[0],
        doc_limit=None,
        top_k=50,
        llm_provider="openrouter",
        llm_model="z-ai/glm-5.1",
        label_tag=None,
        experiment="default",
    )

    assert seen == {"provider": "openrouter", "model": "z-ai/glm-5.1"}
    assert calls == 1
    assert cost == pytest.approx((500 * 1.23 + 50 * 4.56) / 1_000_000)


def test_runtime_generation_cost_uses_openrouter_glm_pricing():
    result = generate_answer(
        question="Q",
        top_k_nodes=[
            {
                "path_text": "Section",
                "text": "Alpha",
                "text_span": "span",
            }
        ],
        cached_caller=_FakeCachedCaller("A", input_tokens=1000, output_tokens=200),
        llm_provider="openrouter",
        llm_model="z-ai/glm-5.1",
    )

    assert result.cost_usd == pytest.approx(
        compute_cost(1000, 200, "openrouter", model="z-ai/glm-5.1")
    )


def test_runtime_judge_cost_uses_openrouter_glm_pricing():
    result = judge_answer(
        question="Q",
        generated_answer="A",
        ground_truth="A",
        cached_caller=_FakeCachedCaller("True", input_tokens=800, output_tokens=120),
        llm_provider="openrouter",
        llm_model="z-ai/glm-5.1",
    )

    assert result.cost_usd == pytest.approx(
        compute_cost(800, 120, "openrouter", model="z-ai/glm-5.1")
    )


def test_runtime_rag_vanilla_cost_uses_openrouter_glm_pricing():
    result = generate_answer_from_text(
        question="Q",
        context="Context",
        cached_caller=_FakeCachedCaller("A", input_tokens=900, output_tokens=150),
        llm_provider="openrouter",
        llm_model="z-ai/glm-5.1",
    )

    assert result.cost_usd == pytest.approx(
        compute_cost(900, 150, "openrouter", model="z-ai/glm-5.1")
    )


def test_baseline_preprocess_stats_use_openrouter_glm_pricing():
    baseline = _StubBaseline(
        processing_dir=Path("."),
        embed_provider="openai",
        cached_caller=SimpleNamespace(),
        llm_provider="openrouter",
        llm_model="z-ai/glm-5.1",
    )

    baseline.track_preprocess_call(
        CacheResult(
            response="ok",
            input_tokens=700,
            output_tokens=90,
            latency_ms=5.0,
            cache_hit=False,
        )
    )

    stats = baseline.get_preprocess_stats()
    assert stats["preprocess_cost_usd"] == pytest.approx(
        compute_cost(700, 90, "openrouter", model="z-ai/glm-5.1")
    )
