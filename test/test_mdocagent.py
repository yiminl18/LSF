"""Tests for the MDocAgent baseline adapter and extractor.

All subprocess calls are mocked — no real API calls or subprocess invocations.
A gated smoke test (MDOCAGENT_E2E=1) exercises the real subprocess path but is
skipped by default (requires install.sh + OPENAI_API_KEY + OPENAI_API_BASE).
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stub_doc_inputs(tmp_pdf: Path | None = None) -> object:
    from agent.baselines.base import DocInputs
    return DocInputs(
        normalized_text="Amazon is a technology company. It operates globally.",
        entries=[],
        section_index={},
        pdf_path=tmp_pdf or Path("/nonexistent/doc.pdf"),
        ground_truth="Amazon",
    )


# ---------------------------------------------------------------------------
# adapter tests
# ---------------------------------------------------------------------------

class TestDocNameFromDocId(unittest.TestCase):
    def test_plain_id(self) -> None:
        from agent.baselines.mdocagent.adapter import _doc_name_from_doc_id
        self.assertEqual(_doc_name_from_doc_id("AMAZON_2015_10K"), "AMAZON_2015_10K")

    def test_strips_pdf_extension(self) -> None:
        from agent.baselines.mdocagent.adapter import _doc_name_from_doc_id
        self.assertEqual(_doc_name_from_doc_id("AMAZON_2015_10K.pdf"), "AMAZON_2015_10K")

    def test_strips_path_prefix(self) -> None:
        from agent.baselines.mdocagent.adapter import _doc_name_from_doc_id
        self.assertEqual(_doc_name_from_doc_id("docs/foo/AMAZON_2015_10K.pdf"), "AMAZON_2015_10K")


class TestUpsertSample(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.data_dir = Path(self._tmp) / "data" / "lsf"
        self.data_dir.mkdir(parents=True)

    def _load(self, name: str) -> list[dict]:
        p = self.data_dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

    def test_creates_samples_json(self) -> None:
        from agent.baselines.mdocagent.adapter import _upsert_sample
        _upsert_sample(
            data_dir=self.data_dir,
            sample_id="0_AMAZON",
            doc_id="AMAZON_2015_10K",
            query_text="What is the company name?",
            n_pages=3,
        )
        samples = self._load("samples.json")
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["id"], "0_AMAZON")
        self.assertEqual(samples[0]["question"], "What is the company name?")
        self.assertEqual(samples[0]["answer"], "")

    def test_creates_retrieval_json_with_all_pages(self) -> None:
        from agent.baselines.mdocagent.adapter import _upsert_sample
        _upsert_sample(
            data_dir=self.data_dir,
            sample_id="0_AMAZON",
            doc_id="AMAZON_2015_10K",
            query_text="What is the company name?",
            n_pages=5,
        )
        retrieval = self._load("sample-with-retrieval-results.json")
        self.assertEqual(len(retrieval), 1)
        record = retrieval[0]
        # Trivial retrieval: all 5 page indices
        self.assertEqual(record["text-top-10-question"], [0, 1, 2, 3, 4])
        self.assertEqual(record["image-top-10-question"], [0, 1, 2, 3, 4])

    def test_upsert_replaces_existing_sample(self) -> None:
        from agent.baselines.mdocagent.adapter import _upsert_sample
        _upsert_sample(
            data_dir=self.data_dir,
            sample_id="0_AMAZON",
            doc_id="AMAZON_2015_10K",
            query_text="Old question",
            n_pages=2,
        )
        _upsert_sample(
            data_dir=self.data_dir,
            sample_id="0_AMAZON",
            doc_id="AMAZON_2015_10K",
            query_text="New question",
            n_pages=4,
        )
        samples = self._load("samples.json")
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["question"], "New question")
        retrieval = self._load("sample-with-retrieval-results.json")
        self.assertEqual(retrieval[0]["text-top-10-question"], [0, 1, 2, 3])

    def test_multiple_samples_accumulate(self) -> None:
        from agent.baselines.mdocagent.adapter import _upsert_sample
        _upsert_sample(self.data_dir, "0_AMAZON", "AMAZON_2015_10K", "Q1", 2)
        _upsert_sample(self.data_dir, "1_GOOGLE", "GOOGLE_2015_10K", "Q2", 3)
        samples = self._load("samples.json")
        self.assertEqual(len(samples), 2)
        ids = {s["id"] for s in samples}
        self.assertEqual(ids, {"0_AMAZON", "1_GOOGLE"})


class TestPrepareInputsNoRealPdf(unittest.TestCase):
    """prepare_inputs graceful fallback when PDF doesn't exist."""

    def test_creates_text_fallback_when_no_pdf(self) -> None:
        from agent.baselines.mdocagent.adapter import prepare_inputs

        with tempfile.TemporaryDirectory() as tmp:
            # Redirect upstream data/tmp dirs into tmp
            tmp_path = Path(tmp)
            with patch(
                "agent.baselines.mdocagent.adapter._upstream_data_dir",
                return_value=tmp_path / "data",
            ), patch(
                "agent.baselines.mdocagent.adapter._upstream_extract_dir",
                return_value=tmp_path / "extract",
            ):
                (tmp_path / "data").mkdir()
                (tmp_path / "extract").mkdir()

                doc_inputs = _make_stub_doc_inputs()  # pdf_path=/nonexistent/doc.pdf
                info = prepare_inputs(
                    doc_inputs,
                    doc_id="TEST_DOC",
                    dataset_name="lsf",
                    query_idx=0,
                    query_text="What is the company name?",
                )

        self.assertEqual(info["sample_id"], "0_TEST_DOC")
        self.assertEqual(info["n_pages"], 1)

    def test_samples_json_layout(self) -> None:
        from agent.baselines.mdocagent.adapter import prepare_inputs

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            extract_dir = tmp_path / "extract"
            data_dir.mkdir()
            extract_dir.mkdir()

            with patch(
                "agent.baselines.mdocagent.adapter._upstream_data_dir",
                return_value=data_dir,
            ), patch(
                "agent.baselines.mdocagent.adapter._upstream_extract_dir",
                return_value=extract_dir,
            ):
                prepare_inputs(
                    _make_stub_doc_inputs(),
                    doc_id="AMAZON_10K",
                    dataset_name="lsf",
                    query_idx=1,
                    query_text="Revenue?",
                )

            samples = json.loads((data_dir / "samples.json").read_text())
            self.assertEqual(samples[0]["id"], "1_AMAZON_10K")
            self.assertEqual(samples[0]["question"], "Revenue?")
            self.assertEqual(samples[0]["doc_id"], "AMAZON_10K")

            retrieval = json.loads(
                (data_dir / "sample-with-retrieval-results.json").read_text()
            )
            r = retrieval[0]
            self.assertIn("text-top-10-question", r)
            self.assertIn("image-top-10-question", r)


# ---------------------------------------------------------------------------
# extractor tests
# ---------------------------------------------------------------------------

class TestMDocAgentExtractorSubprocess(unittest.TestCase):
    """Test extractor Option B subprocess path with mocked subprocess.run."""

    def _make_extractor(self) -> object:
        from agent.baselines.mdocagent.extractor import MDocAgentExtractor
        return MDocAgentExtractor()

    def _make_caller(self) -> MagicMock:
        from core.pipeline.e2e_utils.cache import CacheResult
        mock = MagicMock()
        mock.call.return_value = CacheResult(
            response="Amazon", input_tokens=10, output_tokens=5,
            latency_ms=10.0, cache_hit=False,
        )
        return mock

    def _make_fake_result_dir(self, tmp: Path, run_name: str, answer: str) -> None:
        """Write a fake MDocAgent result JSON in the expected location."""
        result_dir = tmp / "results" / "lsf" / run_name
        result_dir.mkdir(parents=True)
        ans_key = f"ans_{run_name}"
        result_file = result_dir / "2025-01-01-00-00.json"
        result_file.write_text(
            json.dumps([
                {
                    "id": "0_TEST_DOC",
                    "question": "What is the company name?",
                    ans_key: answer,
                }
            ], indent=2),
            encoding="utf-8",
        )

    def test_subprocess_override_list(self) -> None:
        """Assert the Hydra override list passed to subprocess."""
        from agent.baselines.mdocagent import extractor as mdoc_ext

        # We need to capture what subprocess.run is called with
        captured_cmd: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured_cmd.append(cmd)
            r = MagicMock()
            r.stdout = ""
            r.stderr = ""
            r.returncode = 0
            return r

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Create fake predict.py and result
            scripts_dir = tmp_path / "scripts"
            scripts_dir.mkdir()
            (scripts_dir / "predict.py").write_text("# fake", encoding="utf-8")

            # Patch _UPSTREAM_DIR to point to tmp
            with patch.object(mdoc_ext, "_UPSTREAM_DIR", tmp_path), \
                 patch.object(mdoc_ext, "_upstream_is_present", return_value=True), \
                 patch("agent.baselines.mdocagent.extractor.generate_lsf_dataset_config"), \
                 patch("agent.baselines.mdocagent.extractor.prepare_inputs",
                       return_value={"sample_id": "0_TEST_DOC", "n_pages": 2,
                                     "data_dir": tmp_path, "extract_path": tmp_path}), \
                 patch("subprocess.run", side_effect=fake_run), \
                 patch.object(mdoc_ext, "_parse_result",
                              return_value=("Amazon", {"run_name": "x", "sample_id": "0_TEST_DOC"})), \
                 patch.object(mdoc_ext, "_LOG_DIR", tmp_path / "logs"):

                extractor = self._make_extractor()
                extractor.extract(
                    query_idx=0,
                    query_text="What is the company name?",
                    doc_id="TEST_DOC",
                    doc_inputs=_make_stub_doc_inputs(),
                    cached_caller=self._make_caller(),
                )

        self.assertTrue(len(captured_cmd) > 0, "subprocess.run not called")
        cmd = captured_cmd[0]
        cmd_str = " ".join(cmd)
        # Must contain dataset=lsf
        self.assertIn("dataset=lsf", cmd_str)
        # Must contain all 4 agent model overrides
        self.assertIn("mdoc_agent.agents.0.model=openai", cmd_str)
        self.assertIn("mdoc_agent.agents.1.model=openai", cmd_str)
        self.assertIn("mdoc_agent.agents.2.model=openai", cmd_str)
        self.assertIn("mdoc_agent.sum_agent.model=openai", cmd_str)
        # Must contain scripts/predict.py
        self.assertIn("predict.py", cmd_str)

    def test_extract_returns_extraction_result(self) -> None:
        """Full extract() call returns ExtractionResult with the answer."""
        from agent.baselines.mdocagent import extractor as mdoc_ext
        from agent.baselines.base import ExtractionResult

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            with patch.object(mdoc_ext, "_upstream_is_present", return_value=True), \
                 patch("agent.baselines.mdocagent.extractor.generate_lsf_dataset_config"), \
                 patch("agent.baselines.mdocagent.extractor.prepare_inputs",
                       return_value={"sample_id": "0_TEST_DOC", "n_pages": 2,
                                     "data_dir": tmp_path, "extract_path": tmp_path}), \
                 patch.object(mdoc_ext, "_run_predict_subprocess",
                              return_value=("stdout output", "", 0)), \
                 patch.object(mdoc_ext, "_parse_result",
                              return_value=("Amazon.com Inc.", {
                                  "run_name": "test-run",
                                  "sample_id": "0_TEST_DOC",
                              })), \
                 patch.object(mdoc_ext, "_LOG_DIR", tmp_path / "logs"):

                extractor = self._make_extractor()
                result = extractor.extract(
                    query_idx=0,
                    query_text="What is the company name?",
                    doc_id="TEST_DOC",
                    doc_inputs=_make_stub_doc_inputs(),
                    cached_caller=self._make_caller(),
                )

        self.assertIsInstance(result, ExtractionResult)
        self.assertEqual(result.generated_answer, "Amazon.com Inc.")
        self.assertIn("run_name", result.trace)

    def test_subprocess_failure_raises_runtime_error(self) -> None:
        from agent.baselines.mdocagent import extractor as mdoc_ext

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            with patch.object(mdoc_ext, "_upstream_is_present", return_value=True), \
                 patch("agent.baselines.mdocagent.extractor.generate_lsf_dataset_config"), \
                 patch("agent.baselines.mdocagent.extractor.prepare_inputs",
                       return_value={"sample_id": "0_TEST_DOC", "n_pages": 2,
                                     "data_dir": tmp_path, "extract_path": tmp_path}), \
                 patch.object(mdoc_ext, "_run_predict_subprocess",
                              return_value=("", "Error: model not found", 1)), \
                 patch.object(mdoc_ext, "_LOG_DIR", tmp_path / "logs"):

                extractor = self._make_extractor()
                with self.assertRaises(RuntimeError) as ctx:
                    extractor.extract(
                        query_idx=0,
                        query_text="What is the company name?",
                        doc_id="TEST_DOC",
                        doc_inputs=_make_stub_doc_inputs(),
                        cached_caller=self._make_caller(),
                    )
                self.assertIn("TEST_DOC", str(ctx.exception))

    def test_not_implemented_when_submodule_absent(self) -> None:
        from agent.baselines.mdocagent.extractor import MDocAgentExtractor

        with patch("agent.baselines.mdocagent.extractor._upstream_is_present",
                   return_value=False):
            extractor = MDocAgentExtractor()
            with self.assertRaises(NotImplementedError) as ctx:
                extractor.extract(
                    query_idx=0,
                    query_text="Q",
                    doc_id="D",
                    doc_inputs=_make_stub_doc_inputs(),
                    cached_caller=self._make_caller(),
                )
            self.assertIn("submodule", str(ctx.exception).lower())


class TestParseResult(unittest.TestCase):
    def test_finds_answer_by_sample_id(self) -> None:
        from agent.baselines.mdocagent.extractor import _parse_result

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_name = "lsf-q0-AMAZON-abc123"
            ans_key = f"ans_{run_name}"
            result_dir = tmp_path / "results" / "lsf" / run_name
            result_dir.mkdir(parents=True)
            result_file = result_dir / "2025-01-01-00-00.json"
            result_file.write_text(
                json.dumps([
                    {"id": "0_AMAZON", "question": "Q?", ans_key: "Amazon.com Inc."}
                ]),
                encoding="utf-8",
            )

            # Patch _UPSTREAM_DIR
            from agent.baselines.mdocagent import extractor as mdoc_ext
            with patch.object(mdoc_ext, "_UPSTREAM_DIR", tmp_path):
                answer, trace = _parse_result(
                    run_name=run_name,
                    sample_id="0_AMAZON",
                    stdout="",
                )

        self.assertEqual(answer, "Amazon.com Inc.")
        self.assertIn("result_file", trace)

    def test_returns_fallback_when_no_result_file(self) -> None:
        from agent.baselines.mdocagent.extractor import _parse_result
        from agent.baselines.mdocagent import extractor as mdoc_ext

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(mdoc_ext, "_UPSTREAM_DIR", Path(tmp)):
                answer, trace = _parse_result(
                    run_name="nonexistent-run",
                    sample_id="0_AMAZON",
                    stdout="",
                )

        self.assertEqual(answer, "Information not found.")
        self.assertIn("error", trace)


# ---------------------------------------------------------------------------
# dataset_config tests
# ---------------------------------------------------------------------------

class TestDatasetConfig(unittest.TestCase):
    def test_generates_lsf_yaml_in_overrides(self) -> None:
        from agent.baselines.mdocagent.dataset_config import (
            generate_lsf_dataset_config,
            _OVERRIDES_DIR,
            _LSF_DATASET_YAML,
        )

        # Patch upstream dir to a temp dir so we don't mutate the submodule
        with tempfile.TemporaryDirectory() as tmp:
            fake_upstream_cfg = Path(tmp) / "config" / "dataset"
            fake_upstream_cfg.mkdir(parents=True)

            from agent.baselines.mdocagent import dataset_config as dc
            with patch.object(dc, "_UPSTREAM_DATASET_CFG_DIR", fake_upstream_cfg):
                result = generate_lsf_dataset_config()

            # File written to upstream
            target = fake_upstream_cfg / "lsf.yaml"
            self.assertTrue(target.exists())
            content = target.read_text(encoding="utf-8")
            self.assertIn("name: lsf", content)
            self.assertIn("defaults:", content)


# ---------------------------------------------------------------------------
# Gated smoke test (MDOCAGENT_E2E=1 required)
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    os.environ.get("MDOCAGENT_E2E") == "1",
    "Set MDOCAGENT_E2E=1 to run end-to-end MDocAgent test (requires install.sh + OpenAI creds)",
)
class TestMDocAgentE2E(unittest.TestCase):
    """Real end-to-end smoke test: one minimal 1-page PDF through MDocAgent subprocess.

    Requirements:
        export MDOCAGENT_E2E=1
        export OPENAI_API_KEY=...
        export OPENAI_API_BASE=...  # Azure endpoint
        cd .../LSF && bash src/agent/baselines/mdocagent/upstream/MDocAgent/install.sh
    """

    def _create_tiny_pdf(self, path: Path) -> None:
        """Create a minimal 1-page PDF for testing."""
        try:
            import pypdfium2 as pdfium  # type: ignore[import]
            # Minimal valid PDF (1 page, empty)
            doc = pdfium.PdfDocument.new()
            doc.new_page(width=595, height=842)
            doc.save(str(path))
            doc.close()
            return
        except ImportError:
            pass
        # Fallback: write a minimal PDF byte string
        pdf_bytes = (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f\r\n"
            b"0000000009 00000 n\r\n0000000058 00000 n\r\n0000000115 00000 n\r\n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF\n"
        )
        path.write_bytes(pdf_bytes)

    def test_e2e_creates_result_file(self) -> None:
        """Write a 1-page PDF sample, run predict.py, assert result file appears."""
        from agent.baselines.base import DocInputs
        from agent.baselines.mdocagent.extractor import MDocAgentExtractor
        from agent.baselines.mdocagent.adapter import _UPSTREAM_DIR

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = Path(f.name)
        try:
            self._create_tiny_pdf(pdf_path)

            doc_inputs = DocInputs(
                normalized_text="Amazon is a technology company.",
                entries=[],
                section_index={},
                pdf_path=pdf_path,
                ground_truth="Amazon",
            )

            from core.pipeline.e2e_utils.cache import CachedLLMCaller, DEFAULT_CACHE_DB_PATH
            cached_caller = CachedLLMCaller(DEFAULT_CACHE_DB_PATH)

            extractor = MDocAgentExtractor()
            result = extractor.extract(
                query_idx=0,
                query_text="What is the company name?",
                doc_id="E2E_TEST_DOC",
                doc_inputs=doc_inputs,
                cached_caller=cached_caller,
            )

            self.assertIsInstance(result.generated_answer, str)
            self.assertNotEqual(result.generated_answer, "")
            self.assertIn("run_name", result.trace)
        finally:
            pdf_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
