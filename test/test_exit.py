"""Unit tests for the EXIT baseline extractor (upstream ExitRAG wrapper).

The LLM zero-shot fallback was removed because it is not part of upstream EXIT.
Tests now cover only the Gemma checkpoint path and ExtractionResult shape.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_FIXTURE_TEXT = (
    "Amazon was incorporated in 1994 by Jeff Bezos. "
    "The company started as an online bookstore. "
    "Over the years it expanded into cloud computing. "
    "AWS is now a major revenue driver for Amazon. "
    "The company is headquartered in Seattle, Washington."
)
_STUB_QUERY = "What is the company name?"
_STUB_DOC_ID = "TEST_DOC"


def _make_cached_caller(response: str = "Amazon") -> MagicMock:
    from core.pipeline.e2e_utils.cache import CacheResult
    mock = MagicMock()
    mock.call.return_value = CacheResult(
        response=response,
        input_tokens=100,
        output_tokens=20,
        latency_ms=50.0,
        cache_hit=True,
    )
    return mock


def _make_doc_inputs(text: str = _FIXTURE_TEXT):
    from agent.baselines.base import DocInputs
    return DocInputs(
        normalized_text=text,
        entries=[],
        section_index={},
        pdf_path=Path("/nonexistent/doc.pdf"),
        ground_truth="Amazon",
    )


class TestExitExtractorGemmaPath(unittest.TestCase):
    """Gemma path: upstream compress_documents called; API reader called once."""

    def _make_mock_compressor(self, compressed: str = "Amazon is a company."):
        mock_rag = MagicMock()
        selections = [True, False, True, False, True]
        mock_rag.compress_documents.return_value = (compressed, selections, [0.9, 0.1, 0.8, 0.2, 0.7])
        return mock_rag

    def test_compress_documents_called(self) -> None:
        from agent.baselines.exit.extractor import ExitExtractor
        import agent.baselines.exit.extractor as mod
        import sys

        mock_rag = self._make_mock_compressor()

        fake_exit_rag = MagicMock()
        fake_exit_rag.Document = MagicMock(side_effect=lambda title, text: MagicMock(title=title, text=text))

        cached_caller = _make_cached_caller("Amazon.com Inc.")

        with patch.object(mod, "_get_compressor", return_value=mock_rag), \
             patch.dict(sys.modules, {"exit_rag": fake_exit_rag}):
            result = ExitExtractor().extract(
                query_idx=0,
                query_text=_STUB_QUERY,
                doc_id=_STUB_DOC_ID,
                doc_inputs=_make_doc_inputs(),
                cached_caller=cached_caller,
                llm_provider="azure",
                llm_model="gpt-5.4-mini",
            )

        mock_rag.compress_documents.assert_called_once()
        self.assertEqual(result.trace["classifier_method"], "gemma_checkpoint")
        self.assertEqual(result.trace["n_sentences_selected"], 3)
        cached_caller.call.assert_called_once()

    def test_empty_compression_falls_back_to_first_n_chars(self) -> None:
        from agent.baselines.exit.extractor import ExitExtractor
        import agent.baselines.exit.extractor as mod
        import sys

        mock_rag = MagicMock()
        mock_rag.compress_documents.return_value = ("", [], [])

        fake_exit_rag = MagicMock()
        fake_exit_rag.Document = MagicMock(side_effect=lambda title, text: MagicMock())

        cached_caller = _make_cached_caller("Amazon")

        with patch.object(mod, "_get_compressor", return_value=mock_rag), \
             patch.dict(sys.modules, {"exit_rag": fake_exit_rag}):
            result = ExitExtractor().extract(
                query_idx=0,
                query_text=_STUB_QUERY,
                doc_id=_STUB_DOC_ID,
                doc_inputs=_make_doc_inputs(),
                cached_caller=cached_caller,
            )

        self.assertIsInstance(result.generated_answer, str)
        self.assertGreater(result.trace["context_chars"], 0)


class TestExtractionResultKeys(unittest.TestCase):
    def test_deployed_row_key_conformance(self) -> None:
        from agent.baselines.base import ExtractionResult
        from agent.baselines.scorer_adapter import score_and_build_row
        from agent.rule_runtime.deploy import DeployedRow
        from agent.rules.range_rule_scorer import RangeRuleScore

        stub_result = ExtractionResult(
            generated_answer="Amazon",
            trace={"n_sentences_total": 5, "n_sentences_selected": 3,
                   "classifier_method": "gemma_checkpoint", "context_chars": 200, "threshold": 0.5},
            cost_usd=0.001,
            latency_ms=80.0,
        )
        stub_score = RangeRuleScore(
            generated_answer="Amazon",
            judge_method="llm_generate_then_llm_judge",
            judge_result=True,
            accuracy=1.0,
            metadata={"judge": {"cost_usd": 0.001}},
        )
        with patch("agent.baselines.scorer_adapter.score_generated_answer", return_value=stub_score):
            row = score_and_build_row(
                query_idx=0,
                doc_id=_STUB_DOC_ID,
                policy="baseline-exit",
                result=stub_result,
                ground_truth="Amazon",
                query_text=_STUB_QUERY,
                cached_caller=_make_cached_caller(),
                llm_provider="azure",
                llm_model="gpt-5.4-mini",
            )
        self.assertEqual(set(row.keys()), set(DeployedRow.__annotations__.keys()))

    def test_extractor_name(self) -> None:
        from agent.baselines.exit.extractor import ExitExtractor
        self.assertEqual(ExitExtractor.name, "exit")

    def test_trace_has_required_keys(self) -> None:
        from agent.baselines.exit.extractor import ExitExtractor
        import agent.baselines.exit.extractor as mod
        import sys

        mock_rag = MagicMock()
        mock_rag.compress_documents.return_value = ("context text", [True, False], [0.9, 0.1])
        fake_exit_rag = MagicMock()
        fake_exit_rag.Document = MagicMock(side_effect=lambda title, text: MagicMock())

        with patch.object(mod, "_get_compressor", return_value=mock_rag), \
             patch.dict(sys.modules, {"exit_rag": fake_exit_rag}):
            result = ExitExtractor().extract(
                query_idx=0,
                query_text=_STUB_QUERY,
                doc_id=_STUB_DOC_ID,
                doc_inputs=_make_doc_inputs(),
                cached_caller=_make_cached_caller(),
            )

        for key in ("n_sentences_total", "n_sentences_selected", "classifier_method",
                    "context_chars", "threshold"):
            self.assertIn(key, result.trace, f"missing trace key: {key}")


if __name__ == "__main__":
    unittest.main()
