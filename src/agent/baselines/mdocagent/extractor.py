"""MDocAgentExtractor — multi-modal multi-agent baseline (arXiv:2503.13964).

This extractor wraps the vendored MDocAgent upstream code. The upstream
submodule must be initialised before this extractor can run end-to-end.

If the submodule is absent, extract() raises NotImplementedError with
instructions for the user.

To unblock:
    git submodule update --init src/agent/baselines/mdocagent/upstream
    # Then install its requirements into your virtual environment.

Once the submodule is present, two integration options exist:
  - Option A (preferred): import the 5-agent pipeline directly.
  - Option B (fallback): invoke as a subprocess and parse output.

The current implementation stubs the upstream import with NotImplementedError
and implements Option B's subprocess path, which can be activated once the
submodule is present by removing the stub check.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from agent.baselines.base import BaselineExtractor, DocInputs, ExtractionResult
from agent.baselines.mdocagent.adapter import prepare_inputs, _DEFAULT_TMP_ROOT
from core.pipeline.e2e_utils.cache import CachedLLMCaller

# Submodule is at upstream/MDocAgent (one level deeper than the original stub assumed).
# Entry point: scripts/predict.py (hydra-based); agent classes in agents/mdoc_agent.py.
# Option A (direct import) requires torch/transformers — see install.sh.
# Option B (subprocess) is the active fallback; cwd set to upstream/MDocAgent.
_UPSTREAM_DIR = Path(__file__).parent / "upstream" / "MDocAgent"
_DEFAULT_DATASET_NAME = "pdfs"


def _upstream_is_present() -> bool:
    """Check if the upstream submodule has been initialised."""
    main_candidates = [
        _UPSTREAM_DIR / "scripts" / "predict.py",
        _UPSTREAM_DIR / "agents" / "mdoc_agent.py",
        _UPSTREAM_DIR / "README.md",
    ]
    return any(p.exists() for p in main_candidates)


class MDocAgentExtractor:
    name: str = "mdocagent"

    def extract(
        self,
        *,
        query_idx: int,
        query_text: str,
        doc_id: str,
        doc_inputs: DocInputs,
        cached_caller: CachedLLMCaller,
        llm_provider: str = "azure",
        llm_model: str = "gpt-4o",
    ) -> ExtractionResult:
        if not _upstream_is_present():
            raise NotImplementedError(
                "MDocAgent upstream submodule not initialised. "
                "Run: git submodule update --init src/agent/baselines/mdocagent/upstream\n"
                "Then install its dependencies and retry."
            )

        t0 = time.perf_counter()

        # Prepare input layout
        dataset_name = _DEFAULT_DATASET_NAME
        doc_dir = prepare_inputs(
            doc_inputs,
            doc_id=doc_id,
            dataset_name=dataset_name,
            tmp_root=_DEFAULT_TMP_ROOT,
        )

        # Try Option A: direct import
        result = _try_option_a(
            query_text=query_text,
            doc_id=doc_id,
            doc_dir=doc_dir,
        )

        if result is None:
            # Fall back to Option B: subprocess invocation
            result = _try_option_b(
                query_text=query_text,
                doc_id=doc_id,
                doc_dir=doc_dir,
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if result is None:
            raise RuntimeError(
                f"MDocAgent failed for doc_id={doc_id}: both Option A (import) "
                "and Option B (subprocess) returned no result."
            )

        return ExtractionResult(
            generated_answer=result.get("answer", "Information not found."),
            trace=result,
            cost_usd=0.0,  # MDocAgent cost is not tracked per-call; document in report
            latency_ms=latency_ms,
        )


def _try_option_a(
    query_text: str,
    doc_id: str,
    doc_dir: Path,
) -> dict[str, Any] | None:
    """Attempt to import agents.mdoc_agent directly (Option A).

    Requires torch/transformers/hydra from upstream install.sh.
    Returns result dict or None if imports fail (e.g., missing deps).
    """
    try:
        # upstream/MDocAgent is on sys.path; agents/ is a package inside it.
        sys.path.insert(0, str(_UPSTREAM_DIR))
        from agents.mdoc_agent import MDocAgent  # type: ignore[import]
        import hydra  # type: ignore[import]
        with hydra.initialize(config_path=str(_UPSTREAM_DIR / "config"), version_base="1.2"):
            cfg = hydra.compose(config_name="base")
        mdoc = MDocAgent(cfg.mdoc_agent)
        # MDocAgent.predict_dataset expects a BaseDataset; single-doc inference is not
        # directly supported — fall through to Option B.
        return None
    except (ImportError, ModuleNotFoundError, AttributeError):
        return None


def _try_option_b(
    query_text: str,
    doc_id: str,
    doc_dir: Path,
) -> dict[str, Any] | None:
    """Invoke MDocAgent's scripts/predict.py as a subprocess (Option B, active fallback).

    cwd is set to upstream/MDocAgent so hydra config resolution works.
    Returns result dict parsed from subprocess stdout, or None on failure.
    """
    import json
    import os
    import tempfile

    # MDocAgent entry point is scripts/predict.py (hydra-based)
    predict_script = _UPSTREAM_DIR / "scripts" / "predict.py"
    if not predict_script.exists():
        return None

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        tmp_out = f.name

    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(predict_script),
                "--config-name",
                "base",
                f"run-name={doc_id}",
                f"+query_override={query_text}",
                f"+output_path={tmp_out}",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            cwd=str(_UPSTREAM_DIR),  # hydra config_path is relative to cwd
        )
        if completed.returncode != 0:
            print(f"[mdocagent] subprocess stderr: {completed.stderr[:500]}")
            return None
        with open(tmp_out, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"[mdocagent] Option B error: {exc}")
        return None
    finally:
        try:
            os.unlink(tmp_out)
        except OSError:
            pass
