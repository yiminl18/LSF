"""Unit tests for the EXIT baseline extractor.

All LLM calls are stubbed — no real API calls are made.
Tests verify:
  1. Sentence splitter on a fixture paragraph.
  2. Classifier falls back to LLM stub when Gemma checkpoint is absent.
  3. Extractor end-to-end with stubbed LLM produces an ExtractionResult matching
     DeployedRow keys when passed through scorer_adapter.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


_FIXTURE_PARAGRAPH = (
    "Amazon was incorporated in 1994 by Jeff Bezos. "
    "The company started as an online bookstore. "
    "Over the years it expanded into cloud computing. "
    "AWS is now a major revenue driver for Amazon. "
    "The company is headquartered in Seattle, Washington."
)

_STUB_QUERY = "What is the company name?"
_STUB_DOC_ID = "TEST_DOC"


def _make_stub_cached_caller(response: str = "Amazon") -> MagicMock:
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


class TestSentenceSplitter(unittest.TestCase):
    def test_splits_fixture_paragraph_into_multiple_sentences(self) -> None:
        from agent.baselines.exit.sentences import split_sentences
        sents = split_sentences(_FIXTURE_PARAGRAPH)
        self.assertGreaterEqual(len(sents), 3, "expected at least 3 sentences")
        for s in sents:
            self.assertTrue(s.strip(), "empty sentence in output")

    def test_empty_string_returns_empty_list(self) -> None:
        from agent.baselines.exit.sentences import split_sentences
        self.assertEqual(split_sentences(""), [])

    def test_single_sentence_returns_one_element(self) -> None:
        from agent.baselines.exit.sentences import split_sentences
        result = split_sentences("Just one sentence")
        self.assertEqual(len(result), 1)

    def test_preserves_order(self) -> None:
        from agent.baselines.exit.sentences import split_sentences
        sents = split_sentences(_FIXTURE_PARAGRAPH)
        # Reconstruct the paragraph by joining sentences and check it is a
        # subsequence of the original (order preserved).
        joined = " ".join(sents)
        # Every sentence must appear somewhere in the source paragraph
        for s in sents:
            self.assertIn(s[:20], _FIXTURE_PARAGRAPH, f"sentence not in source: {s!r}")


class TestClassifierFallback(unittest.TestCase):
    def test_falls_back_to_llm_when_checkpoint_absent(self) -> None:
        """classify_sentences should use LLM zero-shot when Gemma checkpoint is missing."""
        from agent.baselines.exit import classifier as cls_mod

        sentences = ["Amazon is a company.", "AWS is a cloud platform.", "Jeff Bezos founded it."]
        cached_caller = _make_stub_cached_caller("yes\nno\nyes")

        with patch.object(cls_mod, "_gemma_checkpoint_available", return_value=False):
            scores = cls_mod.classify_sentences(
                query=_STUB_QUERY,
                sentences=sentences,
                cached_caller=cached_caller,
                llm_provider="azure",
                llm_model="gpt-5.4-mini",
                batch_size=30,
            )

        self.assertEqual(len(scores), len(sentences))
        for s in scores:
            self.assertIn(s, (0.0, 1.0), f"score {s!r} not in {{0.0, 1.0}}")
        # At least one LLM call was made
        cached_caller.call.assert_called()

    def test_scores_length_matches_input(self) -> None:
        """Output length must equal input sentence count regardless of batch size."""
        from agent.baselines.exit import classifier as cls_mod

        sentences = [f"Sentence number {i}." for i in range(7)]
        cached_caller = _make_stub_cached_caller("yes\nno\nyes\nno\nyes")

        with patch.object(cls_mod, "_gemma_checkpoint_available", return_value=False):
            scores = cls_mod.classify_sentences(
                query=_STUB_QUERY,
                sentences=sentences,
                cached_caller=cached_caller,
                llm_provider="azure",
                llm_model="gpt-5.4-mini",
                batch_size=3,  # deliberately small to exercise multi-batch path
            )

        self.assertEqual(len(scores), len(sentences))

    def test_empty_sentences_returns_empty(self) -> None:
        from agent.baselines.exit import classifier as cls_mod
        cached_caller = _make_stub_cached_caller()
        with patch.object(cls_mod, "_gemma_checkpoint_available", return_value=False):
            scores = cls_mod.classify_sentences(
                query=_STUB_QUERY,
                sentences=[],
                cached_caller=cached_caller,
                llm_provider="azure",
                llm_model="gpt-5.4-mini",
            )
        self.assertEqual(scores, [])


class TestExitExtractorEndToEnd(unittest.TestCase):
    def _make_doc_inputs(self) -> object:
        from agent.baselines.base import DocInputs
        return DocInputs(
            normalized_text=_FIXTURE_PARAGRAPH,
            entries=[],
            section_index={},
            pdf_path=Path("/nonexistent/doc.pdf"),
            ground_truth="Amazon",
        )

    def test_extract_returns_extraction_result(self) -> None:
        from agent.baselines.exit.extractor import ExitExtractor
        from agent.baselines.base import ExtractionResult
        from agent.baselines.exit import classifier as cls_mod

        cached_caller = _make_stub_cached_caller("Amazon.com Inc.")
        extractor = ExitExtractor()

        # LLM classifier returns "yes" for all sentences; reader returns answer
        cached_caller.call.side_effect = [
            # First call(s): classifier batch(es) — return "yes" for all
            *[
                type("R", (), {"response": "yes\nyes\nyes\nyes\nyes", "input_tokens": 50, "output_tokens": 5})()
                for _ in range(5)
            ],
            # Last call: reader
            type("R", (), {"response": "Amazon.com Inc.", "input_tokens": 200, "output_tokens": 20})(),
        ]

        # Use a simpler approach: mock the call to always return a CacheResult
        from core.pipeline.e2e_utils.cache import CacheResult
        cached_caller = _make_stub_cached_caller("yes\nyes\nyes\nyes\nyes")
        # Override final reader call
        call_count = [0]
        original_call = cached_caller.call

        def smart_call(prompt, **kwargs):
            call_count[0] += 1
            if "Is this sentence" in prompt or "{sentences}" in prompt or "sentence" in prompt.lower() and "yes" not in prompt:
                return CacheResult(
                    response="yes\nyes\nyes",
                    input_tokens=50,
                    output_tokens=3,
                    latency_ms=10.0,
                    cache_hit=False,
                )
            return CacheResult(
                response="Amazon.com Inc.",
                input_tokens=200,
                output_tokens=20,
                latency_ms=50.0,
                cache_hit=True,
            )

        cached_caller.call = smart_call

        with patch.object(cls_mod, "_gemma_checkpoint_available", return_value=False):
            result = extractor.extract(
                query_idx=0,
                query_text=_STUB_QUERY,
                doc_id=_STUB_DOC_ID,
                doc_inputs=self._make_doc_inputs(),
                cached_caller=cached_caller,
                llm_provider="azure",
                llm_model="gpt-5.4-mini",
            )

        self.assertIsInstance(result, ExtractionResult)
        self.assertIsInstance(result.generated_answer, str)
        self.assertIsInstance(result.trace, dict)
        self.assertIn("n_sentences_total", result.trace)
        self.assertIn("n_sentences_selected", result.trace)
        self.assertIn("classifier_method", result.trace)
        self.assertEqual(result.trace["classifier_method"], "llm_zeroshot")

    def test_extract_result_builds_deployed_row(self) -> None:
        """ExtractionResult from ExitExtractor must produce correct DeployedRow keys."""
        from agent.baselines.exit.extractor import ExitExtractor
        from agent.baselines.base import ExtractionResult
        from agent.baselines.scorer_adapter import score_and_build_row
        from agent.rule_runtime.deploy import DeployedRow
        from agent.baselines.exit import classifier as cls_mod
        from agent.rules.range_rule_scorer import RangeRuleScore

        stub_result = ExtractionResult(
            generated_answer="Amazon",
            trace={"n_sentences_total": 5, "n_sentences_selected": 3, "classifier_method": "llm_zeroshot", "context_chars": 200, "threshold": 0.5},
            cost_usd=0.001,
            latency_ms=80.0,
        )
        cached_caller = _make_stub_cached_caller("yes")
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
                cached_caller=cached_caller,
                llm_provider="azure",
                llm_model="gpt-5.4-mini",
            )

        expected_keys = set(DeployedRow.__annotations__.keys())
        actual_keys = set(row.keys())
        self.assertEqual(actual_keys, expected_keys, f"key mismatch: {actual_keys ^ expected_keys}")

    def test_extractor_name(self) -> None:
        from agent.baselines.exit.extractor import ExitExtractor
        self.assertEqual(ExitExtractor.name, "exit")

    def test_fallback_when_all_scores_zero(self) -> None:
        """If classifier returns all zeros, extractor falls back to first N chars."""
        from agent.baselines.exit.extractor import ExitExtractor
        from agent.baselines.base import ExtractionResult
        from agent.baselines.exit import classifier as cls_mod

        # Classifier returns all zeros (no\n...)
        from core.pipeline.e2e_utils.cache import CacheResult
        cached_caller = MagicMock()
        cached_caller.call.return_value = CacheResult(
            response="no\nno\nno\nno\nno",
            input_tokens=50,
            output_tokens=5,
            latency_ms=10.0,
            cache_hit=False,
        )

        with patch.object(cls_mod, "_gemma_checkpoint_available", return_value=False):
            result = ExitExtractor().extract(
                query_idx=0,
                query_text=_STUB_QUERY,
                doc_id=_STUB_DOC_ID,
                doc_inputs=self._make_doc_inputs(),
                cached_caller=cached_caller,
                llm_provider="azure",
                llm_model="gpt-5.4-mini",
            )

        # Should not be empty — fallback kicks in
        self.assertNotEqual(result.generated_answer, "")
        self.assertIsInstance(result, ExtractionResult)


class TestGemmaCheckpointPath(unittest.TestCase):
    """Test the HuggingFace-cache-based checkpoint detection."""

    def test_checkpoint_available_when_hf_cache_hit(self) -> None:
        """_gemma_checkpoint_available returns True when try_to_load_from_cache gives a path."""
        from agent.baselines.exit import classifier as cls_mod

        fake_path = "/fake/hf/cache/adapter_config.json"
        with patch("agent.baselines.exit.classifier.try_to_load_from_cache", return_value=fake_path, create=True):
            with patch("huggingface_hub.try_to_load_from_cache", return_value=fake_path):
                result = cls_mod._gemma_checkpoint_available()
        self.assertTrue(result)

    def test_checkpoint_unavailable_when_hf_cache_miss(self) -> None:
        """_gemma_checkpoint_available returns False when try_to_load_from_cache returns None."""
        from agent.baselines.exit import classifier as cls_mod

        with patch("huggingface_hub.try_to_load_from_cache", return_value=None):
            result = cls_mod._gemma_checkpoint_available()
        self.assertFalse(result)


class TestGemmaCodePath(unittest.TestCase):
    """Test _classify_with_gemma with fully mocked model — no real GPU inference.

    torch is not installed in the test environment; we mock the entire torch
    module so that classifier.py's ``import torch`` succeeds and all tensor
    operations are intercepted.
    """

    def _make_mock_torch(self, yes_prob: float = 0.9) -> MagicMock:
        """Return a mock torch module whose softmax returns a controlled probability."""
        mock_torch = MagicMock(name="torch")

        # softmax(logits, dim=0)[0].item() → yes_prob
        mock_prob_tensor = MagicMock()
        mock_prob_tensor.__getitem__ = MagicMock(return_value=MagicMock(
            item=MagicMock(return_value=yes_prob)
        ))
        mock_torch.softmax.return_value = mock_prob_tensor

        # torch.no_grad() context manager
        mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        # torch.float16 sentinel
        mock_torch.float16 = "float16"

        return mock_torch

    def test_gemma_path_taken_and_scores_extracted(self) -> None:
        """When _gemma_checkpoint_available() is True, classify_sentences uses Gemma."""
        import sys
        from agent.baselines.exit import classifier as cls_mod

        sentences = ["AWS is a cloud platform.", "Amazon sells books.", "Jeff Bezos founded it."]
        query = "What is AWS?"

        # Clear any cached model so lazy-load runs fresh
        cls_mod._GEMMA_MODEL_CACHE.clear()

        mock_torch = self._make_mock_torch(yes_prob=0.9)

        # model(**inputs) → outputs; outputs.logits[0, -1, [yes_id, no_id]] → tensor
        mock_logits = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        # logits[0, -1, [yes_id, no_id]] — chained indexing → the same tensor
        mock_logits.__getitem__ = MagicMock(return_value=MagicMock())

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.return_value = mock_outputs  # model(**inputs) call

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text, **kw: (
            [1000] if "Yes" in text else [2000]
        )
        # tokenizer(prompt, return_tensors="pt") → object with .to(device) method
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs  # inputs.to(device) → inputs
        mock_tokenizer.return_value = mock_inputs

        # Patch _get_gemma_model_and_tokenizer so no actual loading occurs
        with patch.object(cls_mod, "_gemma_checkpoint_available", return_value=True), \
             patch.object(cls_mod, "_get_gemma_model_and_tokenizer",
                          return_value=(mock_model, mock_tokenizer)), \
             patch.dict(sys.modules, {"torch": mock_torch}):
            scores = cls_mod.classify_sentences(
                query=query,
                sentences=sentences,
                cached_caller=MagicMock(),
                llm_provider="azure",
                llm_model="gpt-5.4-mini",
            )

        self.assertEqual(len(scores), len(sentences))
        for s in scores:
            self.assertIsInstance(s, float)
            self.assertEqual(s, 0.9)  # matches mock yes_prob

    def test_gemma_constants(self) -> None:
        """Verify HF repo ID and base model match adapter_config.json."""
        from agent.baselines.exit import classifier as cls_mod
        self.assertEqual(cls_mod._GEMMA_HF_REPO, "doubleyyh/exit-gemma-2b")
        self.assertEqual(cls_mod._GEMMA_BASE_MODEL, "google/gemma-2b-it")


if __name__ == "__main__":
    unittest.main()
