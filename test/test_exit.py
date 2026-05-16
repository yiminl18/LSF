"""Unit tests for the EXIT baseline extractor (upstream ExitRAG wrapper).

Tests cover:
  1. Gemma checkpoint detection (_gemma_checkpoint_available).
  2. ExitExtractor Gemma path — patches _get_compressor; verifies compress_documents called.
  3. ExitExtractor LLM fallback path — verifies LLM calls and cost accumulation.
  4. _llm_compress sentence splitting and score parsing.
  5. ExtractionResult trace keys and DeployedRow key conformance.
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


class TestGemmaCheckpointDetection(unittest.TestCase):
    def test_available_when_hf_cache_hit(self) -> None:
        from agent.baselines.exit.extractor import _gemma_checkpoint_available
        with patch("agent.baselines.exit.extractor.try_to_load_from_cache" if False else
                   "huggingface_hub.try_to_load_from_cache",
                   return_value="/fake/path/adapter_config.json"):
            # Patch the import inside the function
            with patch("builtins.__import__", wraps=__import__) as mock_import:
                pass  # just checking the function exists
        # Direct functional test: mock the huggingface_hub import inside the function
        import agent.baselines.exit.extractor as mod
        with patch.object(mod, "_gemma_checkpoint_available", wraps=mod._gemma_checkpoint_available):
            # Simulate cache hit by patching the internal import
            import sys
            fake_hf = MagicMock()
            fake_hf.try_to_load_from_cache.return_value = "/fake/adapter_config.json"
            with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
                result = mod._gemma_checkpoint_available()
        self.assertTrue(result)

    def test_unavailable_when_hf_cache_miss(self) -> None:
        import agent.baselines.exit.extractor as mod
        import sys
        fake_hf = MagicMock()
        fake_hf.try_to_load_from_cache.return_value = None
        with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
            result = mod._gemma_checkpoint_available()
        self.assertFalse(result)

    def test_unavailable_when_import_fails(self) -> None:
        import agent.baselines.exit.extractor as mod
        import sys
        with patch.dict(sys.modules, {"huggingface_hub": None}):
            result = mod._gemma_checkpoint_available()
        self.assertFalse(result)


class TestExitExtractorGemmaPath(unittest.TestCase):
    """Gemma path: upstream compress_documents called; API reader called once."""

    def _make_mock_compressor(self, compressed: str = "Amazon is a company."):
        mock_rag = MagicMock()
        # compress_documents returns (compressed_text, selections, scores)
        selections = [True, False, True, False, True]
        mock_rag.compress_documents.return_value = (compressed, selections, [0.9, 0.1, 0.8, 0.2, 0.7])
        return mock_rag

    def test_compress_documents_called(self) -> None:
        from agent.baselines.exit.extractor import ExitExtractor
        import agent.baselines.exit.extractor as mod
        import sys

        mock_rag = self._make_mock_compressor()

        # Fake exit_rag module in sys.modules so the import inside extract() works
        fake_exit_rag = MagicMock()
        fake_exit_rag.Document = MagicMock(side_effect=lambda title, text: MagicMock(title=title, text=text))

        cached_caller = _make_cached_caller("Amazon.com Inc.")

        with patch.object(mod, "_gemma_checkpoint_available", return_value=True), \
             patch.object(mod, "_UPSTREAM_DIR", mod._UPSTREAM_DIR), \
             patch.object(mod, "_get_compressor", return_value=mock_rag), \
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
        self.assertEqual(result.trace["n_sentences_selected"], 3)  # 3 True in selections
        cached_caller.call.assert_called_once()  # reader called exactly once

    def test_empty_compression_falls_back_to_first_n_chars(self) -> None:
        from agent.baselines.exit.extractor import ExitExtractor
        import agent.baselines.exit.extractor as mod
        import sys

        mock_rag = MagicMock()
        mock_rag.compress_documents.return_value = ("", [], [])

        fake_exit_rag = MagicMock()
        fake_exit_rag.Document = MagicMock(side_effect=lambda title, text: MagicMock())

        cached_caller = _make_cached_caller("Amazon")

        with patch.object(mod, "_gemma_checkpoint_available", return_value=True), \
             patch.object(mod, "_get_compressor", return_value=mock_rag), \
             patch.dict(sys.modules, {"exit_rag": fake_exit_rag}):
            result = ExitExtractor().extract(
                query_idx=0,
                query_text=_STUB_QUERY,
                doc_id=_STUB_DOC_ID,
                doc_inputs=_make_doc_inputs(),
                cached_caller=cached_caller,
            )

        # Should still produce an answer (fell back to first N chars)
        self.assertIsInstance(result.generated_answer, str)
        self.assertGreater(result.trace["context_chars"], 0)


class TestExitExtractorLLMFallback(unittest.TestCase):
    """LLM fallback path: called when Gemma checkpoint absent."""

    def test_llm_calls_made_when_no_checkpoint(self) -> None:
        from agent.baselines.exit.extractor import ExitExtractor
        import agent.baselines.exit.extractor as mod

        # cached_caller returns "yes\nno\nyes\nno\nyes" for classifier, "Amazon" for reader
        from core.pipeline.e2e_utils.cache import CacheResult
        call_count = {"n": 0}

        def side_effect(prompt, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:  # classifier call
                return CacheResult("yes\nno\nyes\nno\nyes", 100, 5, 10.0, False)
            return CacheResult("Amazon", 100, 20, 50.0, True)  # reader call

        cached_caller = MagicMock()
        cached_caller.call.side_effect = side_effect

        with patch.object(mod, "_gemma_checkpoint_available", return_value=False), \
             patch.object(mod, "compute_cost", return_value=0.01):
            result = ExitExtractor().extract(
                query_idx=0,
                query_text=_STUB_QUERY,
                doc_id=_STUB_DOC_ID,
                doc_inputs=_make_doc_inputs(),
                cached_caller=cached_caller,
                llm_provider="azure",
                llm_model="gpt-5.4-mini",
            )

        self.assertEqual(result.trace["classifier_method"], "llm_zeroshot")
        self.assertGreaterEqual(cached_caller.call.call_count, 2)  # classifier + reader
        self.assertIsInstance(result.generated_answer, str)

    def test_require_gemma_disables_llm_fallback(self) -> None:
        from agent.baselines.exit.extractor import ExitExtractor
        import agent.baselines.exit.extractor as mod

        cached_caller = _make_cached_caller()

        with patch.dict("os.environ", {"LSF_EXIT_REQUIRE_GEMMA": "1"}), \
             patch.object(mod, "_gemma_checkpoint_available", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                ExitExtractor().extract(
                    query_idx=0,
                    query_text=_STUB_QUERY,
                    doc_id=_STUB_DOC_ID,
                    doc_inputs=_make_doc_inputs(),
                    cached_caller=cached_caller,
                    llm_provider="azure",
                    llm_model="gpt-5.4-mini",
                )

        self.assertIn("required but unavailable", str(ctx.exception))
        cached_caller.call.assert_not_called()

    def test_classifier_cost_accumulates(self) -> None:
        from agent.baselines.exit.extractor import ExitExtractor, _llm_compress
        import agent.baselines.exit.extractor as mod

        from core.pipeline.e2e_utils.cache import CacheResult
        cached_caller = MagicMock()
        cached_caller.call.return_value = CacheResult("yes\nno\nyes", 50, 5, 10.0, False)

        with patch.object(mod, "compute_cost", return_value=0.05):
            _text, _nt, _ns, cost = _llm_compress(
                query=_STUB_QUERY,
                text=_FIXTURE_TEXT,
                threshold=0.5,
                cached_caller=cached_caller,
                llm_provider="azure",
                llm_model="gpt-5.4-mini",
            )

        # cost = n_batches * 0.05
        self.assertGreater(cost, 0.0)
        self.assertAlmostEqual(cost, 0.05 * cached_caller.call.call_count, places=6)


class TestLLMCompress(unittest.TestCase):
    """Unit tests for _llm_compress helper."""

    def test_empty_text_returns_empty(self) -> None:
        from agent.baselines.exit.extractor import _llm_compress
        compressed, n_total, n_selected, cost = _llm_compress(
            query="q", text="", threshold=0.5,
            cached_caller=MagicMock(), llm_provider="azure", llm_model="m",
        )
        self.assertEqual(n_total, 0)
        self.assertEqual(cost, 0.0)

    def test_all_no_returns_fallback(self) -> None:
        from agent.baselines.exit.extractor import _llm_compress
        import agent.baselines.exit.extractor as mod
        from core.pipeline.e2e_utils.cache import CacheResult

        cached_caller = MagicMock()
        cached_caller.call.return_value = CacheResult("no\nno\nno\nno\nno", 50, 5, 10.0, False)

        with patch.object(mod, "compute_cost", return_value=0.0):
            compressed, n_total, n_selected, _cost = _llm_compress(
                query=_STUB_QUERY,
                text=_FIXTURE_TEXT,
                threshold=0.5,
                cached_caller=cached_caller,
                llm_provider="azure",
                llm_model="gpt-5.4-mini",
            )

        self.assertGreater(len(compressed), 0, "fallback should return non-empty text")
        self.assertEqual(n_selected, 0)


class TestExtractionResultKeys(unittest.TestCase):
    def test_deployed_row_key_conformance(self) -> None:
        from agent.baselines.exit.extractor import ExitExtractor
        from agent.baselines.base import ExtractionResult
        from agent.baselines.scorer_adapter import score_and_build_row
        from agent.rule_runtime.deploy import DeployedRow
        from agent.rules.range_rule_scorer import RangeRuleScore

        stub_result = ExtractionResult(
            generated_answer="Amazon",
            trace={"n_sentences_total": 5, "n_sentences_selected": 3,
                   "classifier_method": "llm_zeroshot", "context_chars": 200, "threshold": 0.5},
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

        with patch.object(mod, "_gemma_checkpoint_available", return_value=True), \
             patch.object(mod, "_get_compressor", return_value=mock_rag), \
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
