"""MDocAgentExtractor — multi-modal multi-agent baseline (arXiv:2503.13964).

Integration approach: Option B (subprocess).
Option A (direct import) requires torch/transformers from install.sh; those
are not in the LSF venv. Option B shells out to ``scripts/predict.py`` via
subprocess with Hydra command-line overrides, keeping dependency isolation.

Pre-requisites for end-to-end use:
    1. git submodule update --init src/agent/baselines/mdocagent/upstream
    2. cd src/agent/baselines/mdocagent/upstream/MDocAgent && bash install.sh
    3. Set OPENAI_API_KEY and OPENAI_API_BASE env vars (Azure endpoint).
    4. Set MDOCAGENT_E2E=1 to enable the optional smoke test.

Hydra overrides passed to predict.py:
    dataset=lsf
    run-name=<unique>
    mdoc_agent.agents.0.model=openai
    mdoc_agent.agents.1.model=openai
    mdoc_agent.agents.2.model=openai
    mdoc_agent.sum_agent.model=openai

The ``openai`` model config (config/model/openai.yaml) uses the standard
``openai.OpenAI`` client. We route it to Azure by setting OPENAI_API_BASE
and OPENAI_API_KEY in the subprocess environment.

DEVIATION from paper (trivial retrieval): we pre-supply all page indices as
the retrieved set in sample-with-retrieval-results.json, bypassing ColBERT.
The 5-agent reasoning pipeline is unchanged.

Result file location:
    upstream/MDocAgent/results/lsf/<run-name>/<timestamp>.json
    (written by BaseDataset.dump_reults via predict_dataset)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from agent.baselines.base import BaselineExtractor, DocInputs, ExtractionResult
from agent.baselines.mdocagent.adapter import (
    _DEFAULT_DATASET_NAME,
    prepare_inputs,
    _doc_name_from_doc_id,
)
from agent.baselines.mdocagent.dataset_config import generate_lsf_dataset_config
from core.pipeline.e2e_utils.cache import CachedLLMCaller

logger = logging.getLogger(__name__)

# Submodule root
_UPSTREAM_DIR = Path(__file__).parent / "upstream" / "MDocAgent"

# Log directory for subprocess output (outside the submodule)
_LOG_DIR = Path(".cache") / "mdocagent" / "logs"

# Hydra override list for OpenAI model on all agents
# Syntax: Hydra 1.2 CLI overrides use key=value (no leading '~')
# Agents list indexing: 0=image_agent, 1=text_agent, 2=general_agent (from base.yaml)
_AGENT_MODEL_OVERRIDES = [
    "mdoc_agent.agents.0.model=openai",
    "mdoc_agent.agents.1.model=openai",
    "mdoc_agent.agents.2.model=openai",
    "mdoc_agent.sum_agent.model=openai",
]


def _upstream_is_present() -> bool:
    """Check if the upstream submodule has been initialised."""
    candidates = [
        _UPSTREAM_DIR / "scripts" / "predict.py",
        _UPSTREAM_DIR / "agents" / "mdoc_agent.py",
        _UPSTREAM_DIR / "README.md",
    ]
    return any(p.exists() for p in candidates)


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
                "Then run install.sh and set OPENAI_API_KEY + OPENAI_API_BASE."
            )

        t0 = time.perf_counter()

        # Generate lsf.yaml dataset config into upstream submodule
        generate_lsf_dataset_config()

        # Prepare data layout (render pages, write samples.json + retrieval JSON)
        info = prepare_inputs(
            doc_inputs,
            doc_id=doc_id,
            dataset_name=_DEFAULT_DATASET_NAME,
            query_idx=query_idx,
            query_text=query_text,
        )

        # Unique run name so parallel runs don't clobber each other
        run_name = f"lsf-q{query_idx}-{_doc_name_from_doc_id(doc_id)}-{uuid.uuid4().hex[:6]}"

        # Run subprocess
        stdout, stderr, returncode = _run_predict_subprocess(run_name)

        # Log output
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = _LOG_DIR / f"{run_name}.log"
        log_file.write_text(
            f"=== stdout ===\n{stdout}\n\n=== stderr ===\n{stderr}\n",
            encoding="utf-8",
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if returncode != 0:
            logger.error(
                "MDocAgent subprocess failed (rc=%d) for doc_id=%s. Log: %s",
                returncode, doc_id, log_file,
            )
            raise RuntimeError(
                f"MDocAgent subprocess exited with code {returncode} for doc_id={doc_id}. "
                f"See {log_file} for details."
            )

        # Parse result
        answer, trace = _parse_result(
            run_name=run_name,
            sample_id=info["sample_id"],
            stdout=stdout,
        )

        return ExtractionResult(
            generated_answer=answer,
            trace=trace,
            cost_usd=0.0,  # MDocAgent doesn't expose per-call token counts
            latency_ms=latency_ms,
        )


def _run_predict_subprocess(run_name: str) -> tuple[str, str, int]:
    """Invoke MDocAgent's scripts/predict.py via subprocess with Hydra overrides.

    Returns (stdout, stderr, returncode).
    """
    predict_script = _UPSTREAM_DIR / "scripts" / "predict.py"

    # Hydra overrides: dataset=lsf swaps in our dataset config; model overrides
    # swap all three actors + sum_agent to use OpenAI API.
    overrides = [
        "dataset=lsf",
        f"run-name={run_name}",
    ] + _AGENT_MODEL_OVERRIDES

    # Build env: pass through current env, add/override API credentials
    env = os.environ.copy()
    # OPENAI_API_KEY and OPENAI_API_BASE should already be set; we don't override
    # them here so the caller controls which Azure endpoint is used.

    # Disable CUDA (we're using OpenAI API — no local GPU needed)
    env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = [sys.executable, str(predict_script)] + overrides

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min per doc
            check=False,
            cwd=str(_UPSTREAM_DIR),
            env=env,
        )
        return completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired:
        return "", "subprocess timed out after 600s", 124


def _parse_result(
    run_name: str,
    sample_id: str,
    stdout: str,
) -> tuple[str, dict[str, Any]]:
    """Read MDocAgent's result JSON and extract the answer for our sample.

    MDocAgent writes results to:
        upstream/MDocAgent/results/lsf/<run-name>/<YYYY-MM-DD-HH-MM>.json

    The result file is a list of sample dicts. Each dict has the answer under
    the key ``ans_<run-name>`` (cfg.mdoc_agent.ans_key = "ans_${run-name}").
    """
    result_dir = _UPSTREAM_DIR / "results" / _DEFAULT_DATASET_NAME / run_name
    ans_key = f"ans_{run_name}"

    result_file = _find_result_file(result_dir)
    if result_file is None:
        logger.warning("No result file found in %s; returning empty answer.", result_dir)
        return "Information not found.", {
            "run_name": run_name,
            "sample_id": sample_id,
            "result_dir": str(result_dir),
            "error": "result file not found",
        }

    try:
        samples = json.loads(result_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to parse result file %s: %s", result_file, exc)
        return "Information not found.", {
            "run_name": run_name,
            "sample_id": sample_id,
            "result_file": str(result_file),
            "error": str(exc),
        }

    # Find our sample by id
    matching = [s for s in samples if s.get("id") == sample_id]
    if not matching:
        # Try matching on doc_id substring (defensive fallback)
        matching = samples  # take first if only one sample

    sample = matching[0] if matching else {}
    answer = sample.get(ans_key) or sample.get("answer") or "Information not found."
    if not isinstance(answer, str):
        answer = str(answer)

    trace: dict[str, Any] = {
        "run_name": run_name,
        "sample_id": sample_id,
        "result_file": str(result_file),
        "ans_key": ans_key,
        "raw_sample": sample,
    }
    # Include per-agent messages if save_message=true was set
    for agent_key in ("general", "critical", "text", "image", "summary"):
        full_key = f"{ans_key}_message"
        if full_key in sample:
            trace[f"{agent_key}_messages"] = sample.get(full_key)

    return answer, trace


def _find_result_file(result_dir: Path) -> Path | None:
    """Find the latest result JSON file in the result directory."""
    if not result_dir.exists():
        return None
    # BaseDataset.dump_reults writes <YYYY-MM-DD-HH-MM>.json (no _results suffix)
    pattern = "????-??-??-??-??.json"
    candidates = [
        f for f in result_dir.glob(pattern)
        if not f.name.endswith("_results.json")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
