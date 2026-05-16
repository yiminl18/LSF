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

    def test_qa_agent_extractor_protocol(self) -> None:
        from agent.baselines.qa_agent.extractor import QAAgentExtractor
        self._assert_extractor_protocol(QAAgentExtractor())


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


class TestQAAgentExtractor(unittest.TestCase):
    def _make_structured_doc_context(self):
        from agent.tool_agent.document import DocumentContext

        entries = [
            {
                "label": "section_header",
                "text": "Company Overview",
                "page_no": 1,
                "structure": {"level": "H1", "parent_id": None},
            },
            {
                "label": "text",
                "text": "Amazon is a technology company.",
                "page_no": 1,
                "structure": {"parent_id": 0},
            },
            {
                "label": "section_header",
                "text": "Cloud Services",
                "page_no": 2,
                "structure": {"level": "H1", "parent_id": None},
            },
            {
                "label": "text",
                "text": "AWS is a cloud platform.",
                "page_no": 2,
                "structure": {"parent_id": 2},
            },
        ]
        return DocumentContext(
            doc_id="DOC",
            query_idx=0,
            normalized_text=(
                "Company Overview\nAmazon is a technology company.\n\n"
                "Cloud Services\nAWS is a cloud platform."
            ),
            ground_truth="Amazon",
            entries=entries,
            section_index={0: entries[0], 2: entries[2]},
        )

    def test_extract_uses_document_tool_then_final_answer(self) -> None:
        from agent.baselines.qa_agent.extractor import QAAgentExtractor
        from core.pipeline.e2e_utils.cache import CacheResult

        cached_caller = MagicMock()
        cached_caller.call.side_effect = [
            CacheResult(
                response=json.dumps(
                    {
                        "action": "tool",
                        "reasoning": "Search for the company name.",
                        "tool": "keyword_search",
                        "args": json.dumps({"query": "company name", "top_k": 1}),
                        "answer": None,
                    }
                ),
                input_tokens=100,
                output_tokens=20,
                latency_ms=10.0,
                cache_hit=True,
            ),
            CacheResult(
                response=json.dumps(
                    {
                        "action": "final",
                        "reasoning": "The searched evidence identifies Amazon.",
                        "tool": None,
                        "args": None,
                        "answer": "Amazon",
                    }
                ),
                input_tokens=120,
                output_tokens=20,
                latency_ms=10.0,
                cache_hit=True,
            ),
        ]

        result = QAAgentExtractor().extract(
            query_idx=_STUB_QUERY_IDX,
            query_text=_STUB_QUERY_TEXT,
            doc_id=_STUB_DOC_ID,
            doc_inputs=_STUB_DOC_INPUTS,
            cached_caller=cached_caller,
            llm_provider="azure",
            llm_model="gpt-5.4-mini",
        )

        self.assertIsInstance(result, ExtractionResult)
        self.assertEqual(result.generated_answer, "Amazon")
        self.assertEqual(result.trace["termination_reason"], "final")
        self.assertEqual(result.trace["tool_calls"][0]["tool"], "keyword_search")
        self.assertEqual(cached_caller.call.call_count, 2)

    def test_qa_agent_extractor_name(self) -> None:
        from agent.baselines.qa_agent.extractor import QAAgentExtractor
        self.assertEqual(QAAgentExtractor.name, "qa-agent")

    def test_qa_agent_embedding_provider_defaults_to_openrouter(self) -> None:
        from agent.baselines.qa_agent.extractor import _resolve_embedding_provider

        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_resolve_embedding_provider(None), "openrouter")

    def test_qa_agent_embedding_model_defaults_to_openrouter_model(self) -> None:
        from agent.baselines.qa_agent.extractor import _resolve_embedding_model

        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                _resolve_embedding_model("openrouter", None),
                "openai/text-embedding-3-small",
            )

    def test_qa_agent_does_not_expose_overview_tool(self) -> None:
        from agent.baselines.qa_agent.extractor import _QAToolRegistry

        registry = _QAToolRegistry(
            self._make_structured_doc_context(),
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )
        result = registry.dispatch("overview", {})

        self.assertFalse(result.success)
        self.assertIn("Unknown tool", result.error or "")
        self.assertNotIn("overview", _QAToolRegistry.TOOL_NAMES)
        self.assertNotIn("search", _QAToolRegistry.TOOL_NAMES)
        self.assertNotIn("grep", _QAToolRegistry.TOOL_NAMES)
        self.assertNotIn("embedding_search", _QAToolRegistry.TOOL_NAMES)

    def test_qa_agent_prompt_uses_anonymous_document_metadata(self) -> None:
        from agent.baselines.qa_agent.extractor import _build_system_prompt
        from agent.tool_agent.document import DocumentContext

        doc = self._make_structured_doc_context()
        sensitive_doc_id = "ACME_sensitive_filename_2026.pdf"
        anonymous_doc = DocumentContext(
            doc_id=sensitive_doc_id,
            query_idx=doc.query_idx,
            normalized_text=doc.normalized_text,
            ground_truth=doc.ground_truth,
            entries=doc.entries,
            section_index=doc.section_index,
        )

        prompt = _build_system_prompt("What is the company name?", anonymous_doc)

        self.assertIn("Anonymous document metadata:", prompt)
        self.assertNotIn(sensitive_doc_id, prompt)
        self.assertNotIn("doc_id=", prompt)
        self.assertNotIn("- overview:", prompt)
        self.assertIn("- semantic_search:", prompt)
        self.assertIn("- keyword_search:", prompt)
        self.assertIn("- read_chunk:", prompt)
        self.assertIn("- read_pages:", prompt)
        self.assertIn("- python:", prompt)

    def test_keyword_search_returns_chunk_hits(self) -> None:
        from agent.baselines.qa_agent.extractor import _QAToolRegistry

        registry = _QAToolRegistry(
            self._make_structured_doc_context(),
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )
        result = registry.dispatch(
            "keyword_search",
            {"query": "cloud platform", "top_k": 1},
        )

        self.assertTrue(result.success)
        self.assertEqual(result.data[0]["chunk_id"], "p2_c0")
        self.assertIn("AWS", result.data[0]["preview"])
        self.assertIn("preview_chars", result.data[0])
        self.assertIn("truncated_chars", result.data[0])
        self.assertIn("is_truncated", result.data[0])

    def test_keyword_search_reports_truncated_preview_chars(self) -> None:
        from agent.baselines.base import DocInputs
        from agent.baselines.qa_agent.extractor import _QAToolRegistry

        long_text = "Court of Appeals docket number " + ("padding " * 40) + "No. 22-10309"
        doc_inputs = DocInputs(
            normalized_text=long_text,
            entries=[],
            section_index={},
            pdf_path=Path("/nonexistent/doc.pdf"),
            ground_truth="",
        )
        from agent.baselines.qa_agent.extractor import _to_document_context

        registry = _QAToolRegistry(
            _to_document_context(0, "DOC", doc_inputs),
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )

        result = registry.dispatch(
            "keyword_search",
            {"query": "Court of Appeals docket number", "top_k": 1},
        )

        self.assertTrue(result.success)
        self.assertTrue(result.data[0]["is_truncated"])
        self.assertGreater(result.data[0]["truncated_chars"], 0)
        self.assertIn("chars omitted", result.data[0]["truncation_note"])
        self.assertEqual(
            result.data[0]["truncated_chars"],
            result.data[0]["total_chars"] - result.data[0]["preview_chars"],
        )

    def test_regex_search_returns_exact_match_context(self) -> None:
        from agent.baselines.qa_agent.extractor import _QAToolRegistry

        registry = _QAToolRegistry(
            self._make_structured_doc_context(),
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )
        result = registry.dispatch(
            "regex_search",
            {"pattern": r"A[A-Z]S", "case_sensitive": True},
        )

        self.assertTrue(result.success)
        self.assertEqual(result.data[0]["chunk_id"], "p2_c0")
        self.assertEqual(result.data[0]["match"], "AWS")
        self.assertIn("cloud platform", result.data[0]["preview"])
        self.assertIn("truncated_chars", result.data[0])

    def test_read_chunk_and_read_pages_return_text(self) -> None:
        from agent.baselines.qa_agent.extractor import _QAToolRegistry

        registry = _QAToolRegistry(
            self._make_structured_doc_context(),
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )

        chunk_result = registry.dispatch("read_chunk", {"chunk_id": "p2_c0"})
        page_result = registry.dispatch("read_pages", {"start": 2, "end": 2})

        self.assertTrue(chunk_result.success)
        self.assertIn("AWS is a cloud platform.", chunk_result.data)
        self.assertTrue(page_result.success)
        self.assertIn("[page 2]", page_result.data)
        self.assertIn("Cloud Services", page_result.data)

    def test_python_tool_runs_restricted_document_analysis(self) -> None:
        from agent.baselines.qa_agent.extractor import _QAToolRegistry

        registry = _QAToolRegistry(
            self._make_structured_doc_context(),
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )
        code = """
import re
from collections import Counter

matches = re.findall(r"\\b[A-Z]{3}\\b", text)
labels = Counter(entry.get("label") for entry in entries)
print("matches", matches)
result = {"matches": matches, "labels": dict(labels)}
"""
        tool_result = registry.dispatch("python", {"code": code})

        self.assertTrue(tool_result.success)
        self.assertIn("STDOUT:", tool_result.data)
        self.assertIn("matches", tool_result.data)
        self.assertIn("RESULT:", tool_result.data)
        self.assertIn('"matches": [', tool_result.data)
        self.assertIn('"section_header": 2', tool_result.data)

    def test_python_tool_rejects_file_io(self) -> None:
        from agent.baselines.qa_agent.extractor import _QAToolRegistry

        registry = _QAToolRegistry(
            self._make_structured_doc_context(),
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )
        tool_result = registry.dispatch("python", {"code": "open('/tmp/x', 'w')"})

        self.assertTrue(tool_result.success)
        self.assertIn("STDERR:", tool_result.data)
        self.assertIn("forbidden name: open", tool_result.data)

    def test_qa_agent_requires_full_read_after_truncated_search_preview(self) -> None:
        from agent.baselines.base import DocInputs
        from agent.baselines.qa_agent.extractor import QAAgentExtractor
        from core.pipeline.e2e_utils.cache import CacheResult

        court_front_matter = (
            "FOR PUBLICATION\n\n"
            "UNITED STATES COURT OF APPEALS\n"
            "FOR THE NINTH CIRCUIT\n\n"
            "UNITED STATES OF AMERICA,\n\n"
            "Plaintiff-Appellee,\n\n"
            "v.\n\n"
            "ROBERT MANNING,\n\n"
            "Defendant-Appellant.\n"
            "No. 22-10309\n\n"
            "D.C. No. 3:19-cr-00313-WHA-1\n\n"
            "UNITED STATES OF AMERICA,\n\n"
            "Plaintiff-Appellee,\n\n"
            "v.\n\n"
            "JAMARE COATS,\n\n"
            "Defendant-Appellant.\n"
            "No. 22-10310\n"
        )
        doc_inputs = DocInputs(
            normalized_text=court_front_matter,
            entries=[],
            section_index={},
            pdf_path=Path("/nonexistent/doc.pdf"),
            ground_truth="",
        )
        cached_caller = MagicMock()
        cached_caller.call.side_effect = [
            CacheResult(
                response=json.dumps(
                    {
                        "action": "tool",
                        "reasoning": "Find docket numbers.",
                        "tool": "keyword_search",
                        "args": json.dumps({"query": "Court of Appeals docket number", "top_k": 1}),
                        "answer": None,
                    }
                ),
                input_tokens=100,
                output_tokens=20,
                latency_ms=10.0,
                cache_hit=True,
            ),
            CacheResult(
                response=json.dumps(
                    {
                        "action": "final",
                        "reasoning": "The preview appears to show the number.",
                        "tool": None,
                        "args": None,
                        "answer": "No. 2",
                    }
                ),
                input_tokens=120,
                output_tokens=20,
                latency_ms=10.0,
                cache_hit=True,
            ),
            CacheResult(
                response=json.dumps(
                    {
                        "action": "tool",
                        "reasoning": "Read the truncated chunk.",
                        "tool": "read_chunk",
                        "args": json.dumps({"chunk_id": "c0"}),
                        "answer": None,
                    }
                ),
                input_tokens=130,
                output_tokens=20,
                latency_ms=10.0,
                cache_hit=True,
            ),
            CacheResult(
                response=json.dumps(
                    {
                        "action": "final",
                        "reasoning": "The full chunk contains both docket numbers.",
                        "tool": None,
                        "args": None,
                        "answer": "22-10309 and 22-10310",
                    }
                ),
                input_tokens=140,
                output_tokens=20,
                latency_ms=10.0,
                cache_hit=True,
            ),
        ]

        result = QAAgentExtractor().extract(
            query_idx=0,
            query_text="What is/are the Court of Appeals docket number(s) for this case?",
            doc_id="DOC",
            doc_inputs=doc_inputs,
            cached_caller=cached_caller,
            llm_provider="azure",
            llm_model="gpt-5.4-mini",
        )

        self.assertEqual(result.generated_answer, "22-10309 and 22-10310")
        self.assertEqual([call["tool"] for call in result.trace["tool_calls"]], ["keyword_search", "read_chunk"])
        self.assertTrue(
            any(
                obs.get("error") == "final_after_truncated_search_without_full_read"
                for obs in result.trace["observations"]
            )
        )
        self.assertEqual(cached_caller.call.call_count, 4)

    def test_embedding_search_uses_cached_chunk_index(self) -> None:
        from agent.baselines.qa_agent import extractor as qa_ext
        from agent.baselines.qa_agent.extractor import _QAToolRegistry

        registry = _QAToolRegistry(
            self._make_structured_doc_context(),
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )

        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch.object(qa_ext, "_EMBED_CACHE_DIR", Path(tmp_dir)), \
             patch("core.embed.embeddings.get_embeddings_batch",
                   return_value=[[1.0, 0.0], [0.0, 1.0]]) as mock_batch, \
             patch("core.embed.embeddings.get_embedding",
                   return_value=[0.0, 1.0]) as mock_query, \
             patch("core.embed.embeddings.get_embedding_cost",
                   side_effect=[(0, 0.0), (0, 0.0), (0, 0.0), (0, 0.0)]):
            result = registry.dispatch(
                "semantic_search",
                {"query": "cloud platform", "top_k": 1},
            )

        self.assertTrue(result.success)
        self.assertEqual(result.data[0]["chunk_id"], "p2_c0")
        self.assertIn("AWS", result.data[0]["preview"])
        mock_batch.assert_called_once()
        mock_query.assert_called_once()

    def test_qa_agent_embedding_cache_path_is_anonymous(self) -> None:
        from agent.baselines.qa_agent.extractor import _embedding_cache_path, _window_chunks

        sensitive_doc_id = "ACME_sensitive_filename_2026.pdf"
        path = _embedding_cache_path(
            doc_id=sensitive_doc_id,
            provider="openai",
            model="text-embedding-3-small",
            chunks=_window_chunks("Some document text."),
        )

        self.assertNotIn(sensitive_doc_id, str(path))
        self.assertNotIn("ACME", str(path))


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

    def test_get_extractor_accepts_qa_agent(self) -> None:
        from agent.baselines import runner

        extractor = runner._get_extractor("qa-agent")
        self.assertEqual(extractor.name, "qa-agent")

    def test_baseline_output_path_includes_dataset(self) -> None:
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
                        {"query_idx": 0, "documents": ["DOC_A", "DOC_B"]}
                    ],
                },
                f,
            )
            config_path = Path(f.name)

        class _QAExtractor:
            name = "qa-agent"

            def __init__(self) -> None:
                self.seen: list[tuple[str, str | None, str | None]] = []

            def extract(self, *, doc_id: str, **kwargs):
                self.seen.append(
                    (
                        doc_id,
                        kwargs.get("embedding_provider"),
                        kwargs.get("embedding_model"),
                    )
                )
                return ExtractionResult(
                    generated_answer="x",
                    trace={},
                    cost_usd=0.0,
                    latency_ms=0.0,
                )

        extractor = _QAExtractor()
        stub_row: dict = {k: None for k in DeployedRow.__annotations__}
        stub_row["actual_cost_usd"] = 0.0
        stub_row["judge_result"] = True

        try:
            with tempfile.TemporaryDirectory() as out_dir, \
                 patch("agent.baselines.runner._get_extractor",
                       return_value=extractor), \
                 patch("agent.baselines.runner.get_query_text",
                       return_value="What is the company?"), \
                 patch("agent.baselines.runner.build_doc_inputs",
                       return_value=_STUB_DOC_INPUTS), \
                 patch("agent.baselines.runner.score_and_build_row",
                       return_value=stub_row), \
                 patch("agent.baselines.runner.summarize_rows",
                       return_value={"deployed_acc": 1.0, "total_cost_usd": 0.0}):
                runner.run_baseline_sweep(
                    experiment="qa-agent",
                    query_indices=[0],
                    config_path=config_path,
                    output_root=Path(out_dir),
                )
                self.assertEqual(
                    extractor.seen,
                    [
                        ("DOC_A", None, None),
                        ("DOC_B", None, None),
                    ],
                )
                rows_path = (
                    Path(out_dir)
                    / "pdfs"
                    / "qa-agent"
                    / "q0"
                    / "baseline_rows.jsonl"
                )
                self.assertTrue(rows_path.exists())
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

        experiments = [
            "baseline-exit",
            "baseline-deepread",
            "baseline-mdocagent",
            "baseline-qa-agent",
        ]
        for exp in experiments:
            with self.subTest(exp=exp):
                with self.assertRaises(SystemExit):
                    parse_args(["--experiment", exp, "--phase", "a"])

    def test_baseline_defaults_to_phase_b(self) -> None:
        from agent.run_pipeline import parse_args

        args = parse_args(["--experiment", "baseline-exit"])
        self.assertEqual(args.experiment, "baseline-exit")
        self.assertEqual(args.phase, "b")

    def test_qa_agent_baseline_phase_b_accepted(self) -> None:
        from agent.run_pipeline import parse_args

        args = parse_args(["--experiment", "baseline-qa-agent", "--phase", "b"])
        self.assertEqual(args.experiment, "baseline-qa-agent")
        self.assertEqual(args.phase, "b")

    def test_qa_agent_accepts_embedding_args(self) -> None:
        from agent.run_pipeline import parse_args

        args = parse_args([
            "--experiment", "baseline-qa-agent",
            "--embed-provider", "openai",
            "--embed-model", "text-embedding-3-small",
        ])
        self.assertEqual(args.embed_provider, "openai")
        self.assertEqual(args.embed_model, "text-embedding-3-small")

    def test_run_pipeline_accepts_llm_and_seed_args(self) -> None:
        from agent.run_pipeline import parse_args

        args = parse_args([
            "--experiment", "baseline-qa-agent",
            "--llm-provider", "openrouter",
            "--llm-model", "openai/gpt-4o-mini",
            "--seed", "7",
        ])
        self.assertEqual(args.llm_provider, "openrouter")
        self.assertEqual(args.llm_model, "openai/gpt-4o-mini")
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.holdout_seed, 7)
        self.assertEqual(args.partition_seed, 7)

    def test_baseline_phase_both_accepted(self) -> None:
        from agent.run_pipeline import parse_args

        args = parse_args(["--experiment", "baseline-exit", "--phase", "both"])
        self.assertEqual(args.experiment, "baseline-exit")
        self.assertEqual(args.phase, "b")

    def test_unknown_experiment_errors(self) -> None:
        from agent.run_pipeline import parse_args

        with self.assertRaises(SystemExit):
            parse_args(["--experiment", "baseline-unknown", "--phase", "b"])


if __name__ == "__main__":
    unittest.main()
