"""Baseline: MDocAgent (arXiv:2503.13964) wrapped to yiming-dev's contract.

Multi-modal multi-agent reader that runs upstream's ``scripts/predict.py`` in a
subprocess, with Hydra overrides pointing every agent at LSF's Azure/OpenAI
adapter. Per-call cost is captured via a JSONL sidecar written by
``baseline.mdocagent.openai_model.MyOpenAI`` and aggregated here.

Public surface ``run_qa(doc_path, question, ...)`` returns the dict shape
``baseline.run_eval`` consumes (``status`` / ``answer`` / ``input_tokens`` /
``output_tokens`` / ``latency_seconds`` / ``total_cost_usd`` / ``model``).

Usage (single pair):
    python src/baseline/agentic_mdocagent.py \\
        --doc data/nopv/raw/<doc>.pdf \\
        --question "On what date was this Notice of Probable Violation issued?" \\
        --model gpt54
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import shutil
import time
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from azure_local import (
    load_azure_credentials_from_key_file,
    load_azure_credentials_from_local,
)
from baseline.mdocagent.adapter import (
    _doc_name_from_doc_id,
    prepare_inputs,
)

# Hydra config name we generate at runtime (see ``dataset_config.generate_lsf_dataset_config``).
# Also used as the shared extract_path subdir and the results subdir lookup.
_UPSTREAM_DATASET_NAME = "lsf"
from baseline.mdocagent.dataset_config import (
    CONFIG_OVERRIDES_ROOT,
    _purge_legacy_upstream_writes,
    generate_lsf_dataset_config,
    generate_lsf_openai_model_config,
    generate_noop_model_config,
)

_UPSTREAM_DIR = _ROOT / "src" / "baseline" / "mdocagent" / "upstream" / "MDocAgent"
_AZURE_JSON = _ROOT / "local" / "azure.json"

# Lets run_eval.py route PDF-only datasets (e.g. nopv) to this baseline.
SUPPORTS_PDF_INPUT = True

_MODEL_ALIASES: dict[str, str] = {
    "gpt54": "gpt-5.4",
    "gpt54mini": "gpt-5.4-mini",
    "gpt-5.4": "gpt-5.4",
    "gpt-5.4-mini": "gpt-5.4-mini",
}

_PDF_DIR_DEFAULTS: dict[str, Path] = {
    "nopv": _ROOT / "data/nopv/raw",
    "financebench": _ROOT / "data/financebench/raw",
}

# Each run_qa call gets its OWN data_dir under upstream/data/run-<run_name>/, so
# samples.json is never shared. The extract_dir (per-page TXT extracts; see
# adapter._extract_pages_from_pdf / _extract_pages_from_parsed_json) IS shared
# and keyed by doc_name; concurrent writes for the same page would just race
# on identical bytes, which is harmless.


def _resolve_model(model: str) -> str:
    return _MODEL_ALIASES.get(model, model)


def _resolve_doc_path(doc_path: Path, pdf_dir: Path | None) -> Path:
    """Resolve to a path the adapter can ingest (``.pdf`` or ``.json``).

    Dispatch:
    - ``.pdf``: returned as-is.
    - ``.json`` that exists on disk: returned as-is (parsed_json source, e.g.
      officeqa's ``datasets/officeqa/latest/parsed_json/<doc>.json``).
    - ``_reconstructed.json`` whose direct path doesn't exist: strip the
      ``_reconstructed`` suffix and look up the matching ``.pdf`` under
      ``pdf_dir`` then ``_PDF_DIR_DEFAULTS`` (financebench-era flow).

    Raises ``FileNotFoundError`` if no candidate resolves.
    """
    if doc_path.suffix.lower() == ".pdf":
        return doc_path

    if doc_path.suffix.lower() == ".json" and doc_path.exists():
        return doc_path

    doc_stem = re.sub(r"_reconstructed$", "", doc_path.stem)

    if pdf_dir is not None:
        candidate = Path(pdf_dir) / f"{doc_stem}.pdf"
        if candidate.exists():
            return candidate

    for default_dir in _PDF_DIR_DEFAULTS.values():
        candidate = default_dir / f"{doc_stem}.pdf"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No PDF or parsed_json found for doc_path={doc_path!r}. Searched: "
        + ", ".join(str(d) for d in ([pdf_dir] if pdf_dir else []) + list(_PDF_DIR_DEFAULTS.values()))
    )


def _model_config_name(model: str) -> str:
    digest = sha256(model.encode("utf-8")).hexdigest()[:8]
    return f"lsf_openai_{digest}"


def _resolve_azure_credentials(resolved_model: str) -> tuple[str, str, str, str | None]:
    """Return (api_key, api_version, endpoint, deployment) for the requested model.

    yiming-dev's convention: ``local/azure.json`` carries the primary (gpt-5.4)
    credentials inline (or via ``key_file``) and an optional ``key_file_cheap``
    pointer for the cheap deployment (gpt-5.4-mini).
    """
    if "mini" in resolved_model.lower():
        cfg = json.loads(_AZURE_JSON.read_text())
        cheap_key_file = cfg.get("key_file_cheap", "")
        if not cheap_key_file:
            raise RuntimeError(
                f"key_file_cheap not set in {_AZURE_JSON}; cannot resolve {resolved_model} credentials"
            )
        return load_azure_credentials_from_key_file(cheap_key_file)
    return load_azure_credentials_from_local(_AZURE_JSON)


def _build_subprocess_env(resolved_model: str, usage_log_path: Path) -> dict[str, str]:
    """Compose env for the upstream predict.py subprocess.

    The Hydra-loaded ``baseline.mdocagent.openai_model.MyOpenAI`` reads these
    variables to bootstrap the Azure client and emit per-call cost telemetry.
    """
    api_key, api_version, endpoint, deployment = _resolve_azure_credentials(resolved_model)
    if not api_key or not endpoint or not api_version:
        raise RuntimeError(
            f"Azure credentials incomplete for model={resolved_model!r}: need api_key, api_version, azure_endpoint"
        )
    if not deployment or not deployment.strip():
        raise RuntimeError(
            f"Azure deployment name missing for model={resolved_model!r}; "
            "the local key_file must include a `deployment:` entry — falling back "
            "to the logical model name produces a misleading 404 from Azure."
        )

    azure_deployment = deployment.strip()

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key
    env["LSF_MDOCAGENT_PROVIDER"] = "azure"
    env["LSF_MDOCAGENT_AZURE_API_BASE"] = endpoint
    env["LSF_MDOCAGENT_AZURE_API_VERSION"] = api_version
    env["LSF_MDOCAGENT_AZURE_DEPLOYMENT"] = azure_deployment
    env["LSF_MDOCAGENT_LOGICAL_MODEL"] = resolved_model
    env["LSF_MDOCAGENT_USAGE_LOG"] = str(usage_log_path)
    env.pop("OPENAI_BASE_URL", None)

    repo_src = str(_ROOT / "src")
    # Upstream MDocAgent must come FIRST so its `models` package wins the
    # `from models.base_model import BaseModel` import inside the subprocess —
    # yiming-dev's src/models/ (gpt54) would otherwise shadow it.
    env["PYTHONPATH"] = (
        str(_UPSTREAM_DIR) + os.pathsep
        + repo_src + os.pathsep
        + env.get("PYTHONPATH", "")
    ).rstrip(os.pathsep)

    if env.get("LSF_MDOCAGENT_PRESERVE_CUDA", "").strip().lower() not in {"1", "true", "yes"}:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _agent_model_overrides(model_config_name: str) -> list[str]:
    # Upstream's default dataset.top_k=1 makes each reader see only page 0;
    # raise to match the "top-10" retrieval keys populated in adapter.py.
    # Also override mdoc_agent.cuda_visible_devices="" — predict.py line 10
    # otherwise resets the env var we cleared in the subprocess shell back to
    # the default "0,1,2,3" before our model code runs.
    #
    # agents.0 (image_agent) is routed to NoOpModel: with retrieval downgraded
    # to BM25 text-only, the image path has no visual signal to add (it would
    # just re-see the same pages as PNG renders). NoOpModel preserves the
    # hardcoded ``self.agents[0]`` index in upstream/agents/mdoc_agent.py while
    # skipping the LLM call. agents.2 (general_agent) keeps the LSF model but
    # switches to a text-only variant — it still drives self_reflect, just
    # without vision input.
    from baseline.mdocagent.adapter import _R_MAX_PAGES as _ADAPTER_MAX_PAGES
    return [
        "mdoc_agent.agents.0.model=noop",
        f"mdoc_agent.agents.1.model={model_config_name}",
        "mdoc_agent.agents.2.agent=general_agent_text_only",
        f"mdoc_agent.agents.2.model={model_config_name}",
        f"mdoc_agent.sum_agent.model={model_config_name}",
        "mdoc_agent.save_message=true",
        # dataset.top_k controls how many retrieved pages each reader actually
        # uses; retrieval.top_k is interpolated into upstream's r_text_key
        # template ``text-top-${retrieval.top_k}-${...}`` so it must match the
        # K our adapter f-strings into _R_TEXT_KEY / _R_IMAGE_KEY. Both flow
        # from _R_MAX_PAGES — change once, propagates everywhere.
        f"dataset.top_k={_ADAPTER_MAX_PAGES}",
        f"retrieval.top_k={_ADAPTER_MAX_PAGES}",
        'mdoc_agent.cuda_visible_devices=""',
    ]


def _run_predict_subprocess(
    run_name: str,
    model_config_name: str,
    data_dir_rel: str,
    env: dict[str, str],
    timeout: int,
) -> tuple[str, str, int]:
    """Invoke upstream's predict.py with Hydra overrides.

    ``data_dir_rel`` is the per-run sample dir as a path relative to the
    upstream root (subprocess cwd), e.g. ``./data/run-<run_name>``.
    """
    predict_script = _UPSTREAM_DIR / "scripts" / "predict.py"
    overrides = [
        "+dataset=lsf",
        f"run-name={run_name}",
        # Per-run sample paths so each subprocess sees exactly its own (doc, q).
        f"dataset.data_dir={data_dir_rel}",
        f"dataset.sample_path={data_dir_rel}/samples.json",
        f"dataset.sample_with_retrieval_path={data_dir_rel}/sample-with-retrieval-results.json",
    ] + _agent_model_overrides(model_config_name)
    # ``--config-dir`` adds our config_overrides to Hydra's GlobalHydra search
    # path *before* the @hydra.main composition runs, so it's also visible to
    # the ``hydra.compose(config_name="model/<name>")`` calls inside predict.py.
    # ``hydra.searchpath=[...]`` only retroactively edits the primary config and
    # is silently ignored by subsequent compose calls — we tried, it doesn't
    # work for our use case.
    cmd = [
        sys.executable, str(predict_script),
        "--config-dir", str(CONFIG_OVERRIDES_ROOT.resolve()),
    ] + overrides
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=str(_UPSTREAM_DIR),
            env=env,
        )
        return completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired:
        return "", f"subprocess timed out after {timeout}s", 124


def _find_result_file(result_dir: Path) -> Path | None:
    if not result_dir.exists():
        return None
    candidates = [
        f for f in result_dir.glob("????-??-??-??-??.json")
        if not f.name.endswith("_results.json")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _parse_result(run_name: str, sample_id: str) -> tuple[str | None, dict[str, Any]]:
    result_dir = _UPSTREAM_DIR / "results" / _UPSTREAM_DATASET_NAME / run_name
    ans_key = f"ans_{run_name}"
    result_file = _find_result_file(result_dir)
    trace: dict[str, Any] = {"run_name": run_name, "sample_id": sample_id, "result_dir": str(result_dir)}
    if result_file is None:
        trace["error"] = "result file not found"
        return None, trace
    trace["result_file"] = str(result_file)
    try:
        samples = json.loads(result_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        trace["error"] = f"failed to parse result file: {exc}"
        return None, trace

    matching = [s for s in samples if s.get("id") == sample_id]
    if not matching:
        trace["error"] = f"sample_id not in result file (available={[s.get('id') for s in samples]})"
        return None, trace

    sample = matching[0]
    raw_answer = sample.get(ans_key) or sample.get("answer")
    trace["ans_key"] = ans_key
    combined_key = f"{ans_key}_message"
    if combined_key in sample:
        trace["combined_messages"] = sample[combined_key]
    if raw_answer is None:
        trace["error"] = "answer field empty"
        return None, trace
    if not isinstance(raw_answer, str):
        raw_answer = str(raw_answer)
    return _compress_answer(raw_answer), trace


def _compress_answer(raw_answer: str) -> str:
    """Trim whitespace around upstream's already-parsed answer.

    ``MultiAgentSystem.sum`` already runs ``json.loads(...).get("Answer")`` and
    stores the result into ``ans_<run-name>``. When that JSON parse succeeds
    we just see the Answer string; when it fails, upstream falls back to the
    raw summarizer response (which may contain prose around the JSON). Either
    way, no additional post-processing here — only whitespace trimming.
    """
    return raw_answer.strip()


def _trim_stderr(stderr: str | None, head: int = 1000, tail: int = 1000) -> str:
    """Keep both the leading Hydra error preamble and the trailing traceback.

    The full subprocess stderr is still written to ``subprocess_log_path``; this
    function only shapes what we put into the returned dict.
    """
    if not stderr:
        return ""
    if len(stderr) <= head + tail + 32:
        return stderr
    omitted = len(stderr) - head - tail
    return f"{stderr[:head]}\n\n... [truncated {omitted} chars] ...\n\n{stderr[-tail:]}"


def _aggregate_usage_log(path: Path) -> tuple[float, int, int, int]:
    if not path.exists():
        return 0.0, 0, 0, 0
    cost = 0.0
    in_tok = 0
    out_tok = 0
    rows = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        cost += float(record.get("cost_usd", 0.0) or 0.0)
        in_tok += int(record.get("input_tokens", 0) or 0)
        out_tok += int(record.get("output_tokens", 0) or 0)
        rows += 1
    return cost, in_tok, out_tok, rows


def _upstream_is_present() -> bool:
    return (_UPSTREAM_DIR / "scripts" / "predict.py").exists()


def _safe_repo_path(path: Path) -> str:
    """Return path relative to repo root, or the absolute path if outside it."""
    try:
        return str(path.relative_to(_ROOT))
    except ValueError:
        return str(path)


def run_qa(
    doc_path: str | Path,
    question: str,
    *,
    model: str = "gpt54",
    timeout: int = 600,
    log_dir: str | Path | None = None,
    log_stem: str | None = None,
    pdf_dir: str | Path | None = None,
    query_idx: int = 0,
    max_pages: int | None = None,
) -> dict:
    """Run MDocAgent on one (doc, question) pair and return the run_eval dict."""
    resolved_model = _resolve_model(model)

    if not _upstream_is_present():
        return {
            "status": "error",
            "answer": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_seconds": 0.0,
            "total_cost_usd": None,
            "model": resolved_model,
            "error_message": (
                "MDocAgent upstream not initialised. Run: "
                "git submodule update --init src/baseline/mdocagent/upstream/MDocAgent"
            ),
        }

    try:
        source = _resolve_doc_path(Path(doc_path), Path(pdf_dir) if pdf_dir else None)
    except FileNotFoundError as exc:
        return {
            "status": "error",
            "answer": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_seconds": 0.0,
            "total_cost_usd": None,
            "model": resolved_model,
            "error_message": str(exc),
        }

    doc_id = source.name
    safe_doc = re.sub(r"[^a-zA-Z0-9_-]", "_", _doc_name_from_doc_id(doc_id))
    run_name = f"lsf-q{query_idx}-{safe_doc}-{uuid.uuid4().hex[:6]}"

    log_base = Path(log_dir) if log_dir is not None else _ROOT / ".cache" / "mdocagent" / "logs"
    log_base.mkdir(parents=True, exist_ok=True)
    stem = log_stem or run_name
    usage_log_path = log_base / f"{stem}.mdocagent.jsonl"
    subprocess_log_path = log_base / f"{stem}.mdocagent.log"
    usage_log_path.write_text("", encoding="utf-8")

    model_config_name = _model_config_name(resolved_model)

    # Per-run private sample dir, shared cache dir for page extracts.
    data_dir = _UPSTREAM_DIR / "data" / f"run-{run_name}"
    extract_dir = _UPSTREAM_DIR / "tmp" / _UPSTREAM_DATASET_NAME
    # Hydra paths are interpreted relative to the subprocess cwd (= _UPSTREAM_DIR),
    # so we hand it the same dir using the upstream-relative form.
    data_dir_rel = f"./data/run-{run_name}"

    t0 = time.time()
    keep_data_dir = False
    try:
        _purge_legacy_upstream_writes()
        generate_lsf_dataset_config()
        generate_noop_model_config()
        generate_lsf_openai_model_config(resolved_model, config_name=model_config_name)
        info = prepare_inputs(
            source,
            doc_id=doc_id,
            data_dir=data_dir,
            extract_dir=extract_dir,
            query_idx=query_idx,
            query_text=question,
            max_pages=max_pages,
        )

        try:
            env = _build_subprocess_env(resolved_model, usage_log_path)
        except Exception as exc:
            keep_data_dir = True
            return {
                "status": "error",
                "answer": None,
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_seconds": round(time.time() - t0, 2),
                "total_cost_usd": None,
                "model": resolved_model,
                "error_message": str(exc),
            }

        stdout, stderr, rc = _run_predict_subprocess(
            run_name, model_config_name, data_dir_rel, env, timeout
        )
        latency = round(time.time() - t0, 2)
        subprocess_log_path.write_text(
            f"=== cmd ===\nrun_name={run_name}\nrc={rc}\n=== stdout ===\n{stdout}\n=== stderr ===\n{stderr}\n",
            encoding="utf-8",
        )

        if rc == 124:
            keep_data_dir = True
            return {
                "status": "timeout",
                "answer": None,
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_seconds": latency,
                "total_cost_usd": None,
                "model": resolved_model,
                "run_name": run_name,
                "mdocagent_log_path": _safe_repo_path(subprocess_log_path),
            }

        answer, parse_trace = _parse_result(run_name=run_name, sample_id=info["sample_id"])
        cost_usd, in_tok, out_tok, gen_calls = _aggregate_usage_log(usage_log_path)

        status = "ok" if rc == 0 and answer is not None else (f"exit_{rc}" if rc != 0 else "no_answer")
        if status != "ok":
            keep_data_dir = True
        return {
            "status": status,
            "answer": answer,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "latency_seconds": latency,
            "total_cost_usd": cost_usd if (in_tok or out_tok) else None,
            "model": resolved_model,
            "gen_calls": gen_calls,
            "n_pages_used": info["n_pages"],
            "retrieval_mode": info.get("retrieval_mode"),
            "retrieved_pages": info.get("retrieved_pages"),
            "run_name": run_name,
            "sample_id": info["sample_id"],
            "mdocagent_log_path": _safe_repo_path(subprocess_log_path),
            "mdocagent_usage_log_path": _safe_repo_path(usage_log_path),
            "mdocagent_trace": parse_trace,
            "stderr_tail": _trim_stderr(stderr),
        }
    finally:
        # Drop both the per-run sample dir AND the per-run result dir on
        # success so they don't accumulate. The result dir contains the same
        # answer we've already parsed into the return value, plus the
        # save_message trace. On failure / timeout we keep both for debugging.
        if not keep_data_dir:
            if data_dir.exists():
                shutil.rmtree(data_dir, ignore_errors=True)
            result_dir = _UPSTREAM_DIR / "results" / _UPSTREAM_DATASET_NAME / run_name
            if result_dir.exists():
                shutil.rmtree(result_dir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="MDocAgent baseline - single pair")
    ap.add_argument("--doc", required=True, help="Path to a PDF or reconstructed JSON")
    ap.add_argument("--question", required=True)
    ap.add_argument("--model", default="gpt54", help="Model alias: gpt54 or gpt54mini")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--pdf-dir", default=None)
    ap.add_argument("--max-pages", type=int, default=None)
    args = ap.parse_args()

    result = run_qa(
        args.doc,
        args.question,
        model=args.model,
        timeout=args.timeout,
        pdf_dir=args.pdf_dir,
        max_pages=args.max_pages,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
