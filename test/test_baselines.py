"""Smoke and protocol-conformance tests for all paper baselines.

All LLM calls are stubbed — no real API calls are made.
Tests verify:
  1. BaselineExtractor protocol conformance.
  2. ExtractionResult key set from each extractor.
  3. DeployedRow key set from scorer_adapter matches rule_runtime.deploy.DeployedRow.
  4. runner.run_baseline_sweep dry_run path.
  5. parse_args rejects --phase a for baseline experiments.
  6. MDocAgent raises NotImplementedError when submodule is absent.
  7. MDocAgent upstream path wired at upstream/MDocAgent/.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from agent.baselines.base import DocInputs, ExtractionResult, BaselineExtractor
from agent.baselines.scorer_adapter import score_and_build_row, error_row
from agent.rule_runtime.deploy import DeployedRow


_STUB_DOC_INPUTS = DocInputs(
    normalized_text="Amazon is a technology company. It operates globally. AWS is a cloud platform.",
    entries=[],
    section_index={},
    pdf_path=Path("/nonexistent/doc.pdf"),
    ground_truth="Amazon",
)

_STUB_QUERY_TEXT = "What is the company name?"
_STUB_QUERY_IDX = 0
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


class TestProtocolConformance(unittest.TestCase):
    """Each extractor must implement BaselineExtractor protocol."""

    def _assert_extractor_protocol(self, extractor: Any) -> None:
        self.assertTrue(hasattr(extractor, "name"), "missing .name attribute")
        self.assertIsInstance(extractor.name, str)
        self.assertTrue(callable(getattr(extractor, "extract", None)), "missing .extract method")

    def test_exit_extractor_protocol(self) -> None:
        from agent.baselines.exit.extractor import ExitExtractor
        self._assert_extractor_protocol(ExitExtractor())

    def test_deepread_extractor_protocol(self) -> None:
        from agent.baselines.deepread.extractor import DeepReadExtractor
        self._assert_extractor_protocol(DeepReadExtractor())

    def test_mdocagent_extractor_protocol(self) -> None:
        from agent.baselines.mdocagent.extractor import MDocAgentExtractor
        self._assert_extractor_protocol(MDocAgentExtractor())


class TestDeepReadExtractor(unittest.TestCase):
    def test_extract_returns_extraction_result_with_fallback_index(self) -> None:
        from agent.baselines.deepread.extractor import DeepReadExtractor
        from agent.baselines.deepread.index import ParagraphIndex

        cached_caller = _make_stub_cached_caller("FINAL ANSWER: Amazon")
        extractor = DeepReadExtractor()

        fixture_index = ParagraphIndex.from_ocr_result({
            "sections": [
                {
                    "section_id": 1,
                    "heading": "Overview",
                    "level": 1,
                    "page_no": 1,
                    "paragraphs": [{"text": "Amazon is the company.", "page_no": 1}],
                }
            ]
        })

        with patch.object(extractor, "_ocr_model", "gpt-4o"), \
             patch("agent.baselines.deepread.extractor.LLMOCR") as MockOCR:
            mock_ocr_instance = MagicMock()
            mock_ocr_instance.parse_pdf.return_value = fixture_index
            MockOCR.return_value = mock_ocr_instance

            result = extractor.extract(
                query_idx=_STUB_QUERY_IDX,
                query_text=_STUB_QUERY_TEXT,
                doc_id=_STUB_DOC_ID,
                doc_inputs=_STUB_DOC_INPUTS,
                cached_caller=cached_caller,
                llm_provider="azure",
                llm_model="gpt-5.4-mini",
            )

        self.assertIsInstance(result, ExtractionResult)
        self.assertIsInstance(result.generated_answer, str)
        self.assertIn("turns", result.trace)

    def test_deepread_extractor_name(self) -> None:
        from agent.baselines.deepread.extractor import DeepReadExtractor
        self.assertEqual(DeepReadExtractor.name, "deepread")


class TestMDocAgentExtractor(unittest.TestCase):
    def test_raises_not_implemented_when_submodule_absent(self) -> None:
        from agent.baselines.mdocagent.extractor import MDocAgentExtractor

        cached_caller = _make_stub_cached_caller()
        extractor = MDocAgentExtractor()

        with patch("agent.baselines.mdocagent.extractor._upstream_is_present", return_value=False):
            with self.assertRaises(NotImplementedError) as ctx:
                extractor.extract(
                    query_idx=_STUB_QUERY_IDX,
                    query_text=_STUB_QUERY_TEXT,
                    doc_id=_STUB_DOC_ID,
                    doc_inputs=_STUB_DOC_INPUTS,
                    cached_caller=cached_caller,
                )
            self.assertIn("submodule", str(ctx.exception).lower())

    def test_mdocagent_extractor_name(self) -> None:
        from agent.baselines.mdocagent.extractor import MDocAgentExtractor
        self.assertEqual(MDocAgentExtractor.name, "mdocagent")

    def test_upstream_path_points_at_mdocagent_subdir(self) -> None:
        """_UPSTREAM_DIR must resolve to upstream/MDocAgent, not upstream/."""
        from agent.baselines.mdocagent import extractor as mdoc_ext
        upstream_dir = mdoc_ext._UPSTREAM_DIR
        # The actual submodule is at upstream/MDocAgent
        self.assertTrue(
            str(upstream_dir).endswith("upstream/MDocAgent"),
            f"_UPSTREAM_DIR should end with 'upstream/MDocAgent', got: {upstream_dir}",
        )


class TestScorerAdapter(unittest.TestCase):
    def test_score_and_build_row_key_set_matches_deployed_row(self) -> None:
        """The output of score_and_build_row must have exactly the same keys as DeployedRow."""
        stub_result = ExtractionResult(
            generated_answer="Amazon",
            trace={},
            cost_usd=0.01,
            latency_ms=100.0,
        )
        cached_caller = _make_stub_cached_caller("yes")

        # Mock score_generated_answer so no LLM call happens
        from agent.rules.range_rule_scorer import RangeRuleScore
        stub_score = RangeRuleScore(
            generated_answer="Amazon",
            judge_method="llm_generate_then_llm_judge",
            judge_result=True,
            accuracy=1.0,
            metadata={"judge": {"cost_usd": 0.001}},
        )

        with patch("agent.baselines.scorer_adapter.score_generated_answer", return_value=stub_score):
            row = score_and_build_row(
                query_idx=_STUB_QUERY_IDX,
                doc_id=_STUB_DOC_ID,
                policy="baseline-exit",
                result=stub_result,
                ground_truth="Amazon",
                query_text=_STUB_QUERY_TEXT,
                cached_caller=cached_caller,
                llm_provider="azure",
                llm_model="gpt-5.4-mini",
            )

        expected_keys = set(DeployedRow.__annotations__.keys())
        actual_keys = set(row.keys())
        self.assertEqual(actual_keys, expected_keys, f"key mismatch: {actual_keys ^ expected_keys}")

    def test_error_row_key_set_matches_deployed_row(self) -> None:
        row = error_row(
            query_idx=0,
            doc_id="DOC",
            policy="baseline-exit",
            blocker="test_error",
        )
        expected_keys = set(DeployedRow.__annotations__.keys())
        self.assertEqual(set(row.keys()), expected_keys)


class TestRunnerDryRun(unittest.TestCase):
    def test_dry_run_does_not_call_extractor(self) -> None:
        from agent.baselines import runner

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            import yaml
            yaml.dump(
                {
                    "dataset": "pdfs",
                    "dataset_root": "datasets/pdfs/latest",
                    "parser": "docling",
                    "queries": [
                        {"query_idx": 0, "documents": ["TEST_DOC_A", "TEST_DOC_B"]}
                    ],
                },
                f,
            )
            config_path = Path(f.name)

        try:
            with patch("agent.baselines.runner._get_extractor") as mock_get, \
                 patch("agent.baselines.runner.get_query_text", return_value="What is the company?"):
                runner.run_baseline_sweep(
                    experiment="exit",
                    query_indices=[0],
                    config_path=config_path,
                    dry_run=True,
                )
                # _get_extractor should still be called to validate the experiment name
                # but no actual extraction should happen.
                mock_get.assert_called_once_with(
                    "exit",
                    deepread_max_pages=None,
                    deepread_ocr_model=None,
                    deepread_ocr_provider=None,
                )
        finally:
            config_path.unlink(missing_ok=True)


class TestRunnerMaxDocs(unittest.TestCase):
    """max_docs param caps the doc list passed to the extractor."""

    def test_max_docs_caps_row_count(self) -> None:
        from agent.baselines import runner

        # Config with 5 docs
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            import yaml
            yaml.dump(
                {
                    "dataset": "pdfs",
                    "dataset_root": "datasets/pdfs/latest",
                    "parser": "docling",
                    "queries": [
                        {
                            "query_idx": 0,
                            "documents": [
                                "DOC_A", "DOC_B", "DOC_C", "DOC_D", "DOC_E"
                            ],
                        }
                    ],
                },
                f,
            )
            config_path = Path(f.name)

        processed: list[str] = []

        class _NoOpExtractor:
            name = "exit"

            def extract(self, *, doc_id: str, **kwargs):
                processed.append(doc_id)
                from agent.baselines.base import ExtractionResult
                return ExtractionResult(
                    generated_answer="x",
                    trace={},
                    cost_usd=0.0,
                    latency_ms=0.0,
                )

        from agent.rule_runtime.deploy import DeployedRow
        stub_row: dict = {k: None for k in DeployedRow.__annotations__}
        stub_row["actual_cost_usd"] = 0.0
        stub_row["judge_result"] = True

        try:
            with tempfile.TemporaryDirectory() as out_dir, \
                 patch("agent.baselines.runner._get_extractor",
                       return_value=_NoOpExtractor()), \
                 patch("agent.baselines.runner.get_query_text",
                       return_value="What is the company?"), \
                 patch("agent.baselines.runner.build_doc_inputs",
                       return_value=_STUB_DOC_INPUTS), \
                 patch("agent.baselines.runner.score_and_build_row",
                       return_value=stub_row), \
                 patch("agent.baselines.runner.summarize_rows",
                       return_value={"deployed_acc": 1.0, "total_cost_usd": 0.0}):
                runner.run_baseline_sweep(
                    experiment="exit",
                    query_indices=[0],
                    config_path=config_path,
                    output_root=Path(out_dir),
                    max_docs=2,
                )
        finally:
            config_path.unlink(missing_ok=True)

        self.assertEqual(len(processed), 2, f"Expected 2 docs processed, got {processed}")
        self.assertEqual(processed, ["DOC_A", "DOC_B"])


class TestRunPipelineParseArgs(unittest.TestCase):
    def test_baseline_phase_a_errors(self) -> None:
        from agent.run_pipeline import parse_args

        experiments = ["baseline-exit", "baseline-deepread", "baseline-mdocagent"]
        for exp in experiments:
            with self.subTest(exp=exp):
                with self.assertRaises(SystemExit):
                    parse_args(["--experiment", exp, "--phase", "a"])

    def test_baseline_phase_b_accepted(self) -> None:
        from agent.run_pipeline import parse_args

        args = parse_args(["--experiment", "baseline-exit", "--phase", "b"])
        self.assertEqual(args.experiment, "baseline-exit")
        self.assertEqual(args.phase, "b")

    def test_baseline_phase_both_accepted(self) -> None:
        from agent.run_pipeline import parse_args

        # --phase both is accepted (main() ignores the phase for baselines and just runs the sweep)
        args = parse_args(["--experiment", "baseline-exit", "--phase", "both"])
        self.assertEqual(args.experiment, "baseline-exit")

    def test_unknown_experiment_errors(self) -> None:
        from agent.run_pipeline import parse_args

        with self.assertRaises(SystemExit):
            parse_args(["--experiment", "baseline-unknown", "--phase", "b"])


if __name__ == "__main__":
    unittest.main()
