"""Evaporate variant runner — subprocess target executed INSIDE `.venv-evaporate`.

**Thin interception over upstream.** This runner drives the *real upstream
Evaporate code* (`evaporate.profiler`, `evaporate.evaluate_profiler`,
`evaporate.profiler_utils`, `evaporate.prompts`) for everything scientifically
meaningful — keyword chunk selection, function synthesis over all generation
templates, noisy-LLM-gold scoring with the 0.5 keep-threshold, top-k selection,
sandboxed function execution, MV/abstention combine — and intercepts only the
four seams that are *absolutely necessary* for a fair, runnable comparison:

  * **N1 — LLM backend.** `evaporate.utils.get_response` is patched to route every
    upstream LLM call through the isolated `llm_backend` (Azure gpt54mini), the
    same backend the rest of the comparison uses. `stop` is never sent to the API
    (gpt-5.4-mini rejects it); it is emulated by local post-truncation.
    Reason: fairness (same backend as LSF) + dependency (manifest dropped) + API.
  * **N2 — weak-supervision engine.** Code+ aggregation runs the modern
    `snorkel.labeling.model.LabelModel` (snorkel 0.10) in place of the original
    Snorkel-MeTaL `LabelModel`, which no longer runs under modern networkx. The
    shim mirrors upstream `weak_supervision/run_ws.py:get_data` label-space
    construction and is installed as `evaporate.profiler.run_ws`, so upstream's
    own `combine_extractions` drives it (and its MV fallbacks). Reason: dep rot.
  * **N3 — WS class prior.** The LabelModel is fit with NO `Y_dev`: a uniform
    `1/k` prior. Upstream's `get_data` estimates the prior from apply-set gold,
    which is a gold leak at apply time. Reason: fairness (absolutely necessary).
  * **N4 — scoring/split/cost.** Final accuracy, the doc split, and cost are owned
    by the orchestrator (`run_evaporate.py`), which reuses pipeline.py's judge,
    split, per-doc schema and two-column cost. (N4 ≠ the upstream function-SELECTION
    scoring, which is internal Evaporate and reused as-is via `evaluate`.)

Everything between those seams is upstream code, unmodified.

Variants
--------
* **direct**    — upstream `get_model_extractions(collecting_preds=True)`: per
                  (question, doc) the LLM reads the keyword-filtered chunks and
                  extracts the span directly. Apply cost = real LLM tokens
                  (phase=extraction). No synthesis.
* **code**      — upstream synthesis (`get_all_extractions`→`get_functions`),
                  scoring (`evaluate`), best-1 selection (`get_topk_scripts_per_field`,
                  k=1), apply to all docs (`apply_final_ensemble`, pure Python),
                  combine (`combine_extractions`, MV-trivial over 1 function).
* **codeplus**  — same, but top-k functions and `combine_extractions` aggregation:
                  combiner=ws → the N2 snorkel `run_ws` shim (with N3 uniform
                  prior); combiner=mv → upstream majority vote.

I/O contract (schema_version=1)
-------------------------------
Input JSON  (``--input``): see `_REQUIRED_INPUT_KEYS`.
Output JSON (``--output``):
    {
      "schema_version": 1,
      "variant": "...",
      "combiner": "ws"|"mv"|null,
      "predictions": { q_slug: { doc_name: {"predicted": str, "extract_llm_tokens": int} } },
      "functions":   { q_slug: [ {"source": str, "score": float, "from_doc": null} ] },
      "codeplus_ws": { q_slug: { ws_applied, mv_fallback, fallback_reasons, ... } },
      "ledger":         [ ... ],
      "ledger_summary": { "by_phase": {synthesis, extraction}, ... },
      "errors":         [ ... ]
    }

`extract_llm_tokens` is the per-doc LLM token spend of the *apply* step (Direct
only; 0 for code/codeplus, whose apply is pure Python). The orchestrator turns
predictions into pipeline-schema per-doc records.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SCHEMA_VERSION = 1
_REQUIRED_INPUT_KEYS = ("schema_version", "variant", "questions", "sampled_docs", "all_docs")

# Code+ weak-supervision (N2/N3). We mirror upstream get_data's per-document,
# rank-based local label space of fixed cardinality: each doc's classes are its
# top-N most common distinct extractions, padded to N with dummies, abstain = -1.
WS_NUM_ELTS = 5          # fixed LabelModel cardinality (upstream get_data num_elts default)

# N1 (model compat): gpt54mini is a reasoning model, so upstream's tiny max_toks
# (10/100) get eaten by hidden reasoning → empty output. Floor the output budget.
_MIN_COMPLETION_TOKENS = 256

_HERE = Path(__file__).resolve().parent
_UPSTREAM = _HERE.parent / "upstream"

# Set by run()/import: the live backend, upstream's value cleaner (for the WS shim),
# and the phase-tagged WS stats the orchestrator reads.
_BACKEND: Any = None
_CLEAN_COMPARISON: Any = None
_WS_STATS_BY_ATTR: dict[str, dict] = {}
# Per-attribute {sampled_doc_name: gold_value} for the N3 WS class prior. Set by
# _run_code before combine_extractions; read by the run_ws shim (which has a fixed
# upstream signature and can't take it as an argument).
_SAMPLED_GOLD_BY_ATTR: dict[str, dict] = {}


# ── N1: LLM backend interception ─────────────────────────────────────────────

def _patched_get_response(prompt, manifest, overwrite=False, max_toks=10,
                          stop_token=None, gold_choices=None, verbose=False):
    """Drop-in for `evaporate.utils.get_response`, routing through `llm_backend`.

    `apply_prompt` looks up `get_response` as a module global at call time, so
    patching `evaporate.utils.get_response` reaches every upstream LLM call even
    though `profiler`/`evaluate_profiler` bound `apply_prompt` into their own
    namespaces at import. We ignore `manifest`, never send `stop` (post-truncate
    locally — N1), and return `(text, total_tokens)` like upstream does.
    """
    # gold_choices is upstream's constrained/log-prob branch; the ClosedIE flow never
    # uses it. Fail loud if that ever changes rather than silently degrading.
    assert gold_choices is None, "gold_choices (constrained scoring) unsupported in the patch"
    text, toks = _BACKEND.complete(
        (prompt or "").strip(),
        max_completion_tokens=max(int(max_toks or 0), _MIN_COMPLETION_TOKENS),
    )
    idx = text.find("---")  # upstream get_response truncates at "---" (utils.py:291)
    if idx != -1:
        text = text[:idx]
    return text.strip(), toks


def _install_stubs() -> None:
    """Neutralize upstream's top-level manifest/metal imports before importing it.

    `utils.py` does `from manifest import Manifest`; `profiler.py` does
    `from evaporate.weak_supervision.run_ws import run_ws` (which drags in
    snorkel-metal/cvxpy). We never use either: LLM calls go through `llm_backend`
    (N1) and WS uses a snorkel shim installed as `profiler.run_ws` (N2). Stub both
    so the imports succeed.
    """
    import types

    if "manifest" not in sys.modules:
        man = types.ModuleType("manifest")

        class _Manifest:  # pragma: no cover - never instantiated
            def __init__(self, *_a, **_k):
                raise RuntimeError("manifest is stubbed (llm_backend in use)")

        man.Manifest = _Manifest
        sys.modules["manifest"] = man

    if "evaporate.weak_supervision.run_ws" not in sys.modules:
        stub = types.ModuleType("evaporate.weak_supervision.run_ws")

        def _run_ws(*_a, **_k):  # pragma: no cover - replaced by the snorkel shim
            raise RuntimeError("upstream metal run_ws is stubbed (snorkel shim installed on profiler)")

        stub.run_ws = _run_ws
        sys.modules["evaporate.weak_supervision.run_ws"] = stub


def _import_upstream():
    """Import upstream pieces, then install the N1 (LLM) and N2 (WS) patches."""
    if str(_UPSTREAM) not in sys.path:
        sys.path.insert(0, str(_UPSTREAM))
    _install_stubs()

    global _CLEAN_COMPARISON

    import evaporate.utils as up_utils  # noqa: E402
    from evaporate.profiler_utils import (  # noqa: E402
        get_txt_parse, filter_file2chunks, clean_function_predictions, set_profiler_args,
    )
    import evaporate.profiler as up_profiler  # noqa: E402
    from evaporate.profiler import (  # noqa: E402
        get_all_extractions, apply_final_ensemble, combine_extractions,
        get_model_extractions, get_function_field_from_attribute,
    )
    from evaporate.evaluate_profiler import evaluate, get_topk_scripts_per_field  # noqa: E402
    from evaporate.evaluate_synthetic import clean_comparison  # noqa: E402

    # The WS shim mirrors upstream get_data, which cleans votes with this exact
    # function (only the empty string is an abstain; "none" stays a vote value).
    _CLEAN_COMPARISON = clean_comparison

    # N1: route all upstream LLM calls through llm_backend.
    up_utils.get_response = _patched_get_response
    # N2: drive Code+ aggregation through the snorkel shim (upstream combine_extractions
    # imported `run_ws` into the profiler namespace — patch it there).
    up_profiler.run_ws = _snorkel_run_ws

    return {
        "get_txt_parse": get_txt_parse,
        "filter_file2chunks": filter_file2chunks,
        "clean_function_predictions": clean_function_predictions,
        "set_profiler_args": set_profiler_args,
        "get_all_extractions": get_all_extractions,
        "apply_final_ensemble": apply_final_ensemble,
        "combine_extractions": combine_extractions,
        "get_model_extractions": get_model_extractions,
        "get_function_field_from_attribute": get_function_field_from_attribute,
        "evaluate": evaluate,
        "get_topk_scripts_per_field": get_topk_scripts_per_field,
    }


# ── N2 + N3: snorkel weak-supervision shim (drop-in for upstream run_ws) ──────

def _snorkel_run_ws(all_votes, gold_extractions_file, symmetric=True, attribute="",
                    has_abstains=1.0, extraction_fraction_thresh=0.9):
    """Snorkel 0.10 LabelModel weak supervision (N2), uniform class prior (N3).

    Same call/return contract as upstream `weak_supervision/run_ws.py:run_ws`:
    returns `(mapped_preds, used_deps, missing_files)` where `mapped_preds` is a
    list aligned to `all_votes` iteration order (one per non-missing doc).
    `combine_extractions` consumes it and applies its own MV fallbacks.

    Faithful to upstream `get_data` for label-space construction (top-`WS_NUM_ELTS`
    rank-based local classes, abstain bucket, dummy padding, `random.seed(0)`
    shuffle, -1 abstain). DEVIATIONS from upstream, by design:
      * N3: the class prior is estimated from the SAMPLED docs' gold only — the
        held-in labels LSF also uses (legitimate parity). Upstream estimates it from
        ALL docs' gold (run_ws.py:76-87), i.e. including the apply set, which is a
        leak; we restrict it to sampled docs. Unsampled docs vote into `L` but carry
        no class-balance label. Sampled gold arrives via `_SAMPLED_GOLD_BY_ATTR`; we
        never read `gold_extractions_file`.
      * N2: snorkel `LabelModel` (0..k-1, -1 abstain) instead of MeTaL (1..k, 0).
      * cvxpy structure-learning/deps dropped — this is upstream's own
        try/except "Not modeling dependencies" fallback path, not a new deviation.
    """
    import random as _random
    import numpy as np

    files = list(all_votes.keys())
    m = max((len(all_votes[f]) for f in files), default=0)
    stats = {"combiner": "ws", "n_docs": len(files), "m_functions": m,
             "cardinality": WS_NUM_ELTS, "ws_applied": 0, "mv_fallback": 0,
             "fallback_reasons": {}}

    def _bump(reason: str, n: int = 1) -> None:
        stats["fallback_reasons"][reason] = stats["fallback_reasons"].get(reason, 0) + n

    has_abs = float(has_abstains) >= float(extraction_fraction_thresh)
    _random.seed(0)  # upstream get_data seeds inside the function

    sampled_gold = _SAMPLED_GOLD_BY_ATTR.get(attribute, {})
    L_rows: list = []
    int_to_name: list[dict[int, str]] = []
    y_dev: list[int] = []                    # N3: class-balance labels, SAMPLED docs only
    for f in files:
        extractions = [_CLEAN_COMPARISON(e) for e in all_votes[f]]
        if has_abs:
            extractions = [e if e else "abstain" for e in extractions]
        unique = [i for i, _ in Counter(extractions).most_common(WS_NUM_ELTS) if i != "abstain"]
        for n in range(WS_NUM_ELTS - len(unique)):
            unique.append(f"dummy{n}")
        _random.shuffle(unique)
        name_to_int = {name: j for j, name in enumerate(unique)}
        int_to_name.append({j: name for name, j in name_to_int.items()})
        L_rows.append([name_to_int.get(a, -1) for a in extractions])
        if f in sampled_gold:
            gv = sampled_gold[f]
            gv = ", ".join(map(str, gv)) if isinstance(gv, list) else str(gv)
            gc = _CLEAN_COMPARISON(gv)
            # gold in this doc's local space → its index; else a random class (upstream get_data:86)
            y_dev.append(name_to_int[gc] if gc in name_to_int else _random.randrange(WS_NUM_ELTS))

    # snorkel's LabelModel needs >=3 functions (MeTaL didn't); below that, MV.
    if m < 3 or len(files) < 2:
        _bump("m<3 (snorkel LabelModel needs >=3 LFs) / <2 docs", len(files))
        stats["mv_fallback"] = len(files)
        _WS_STATS_BY_ATTR[attribute] = stats
        return ["" for _ in files], False, []

    L = np.array(L_rows, dtype=int)
    observed = {int(x) for x in L[L >= 0].tolist()}
    if len(observed) < 2:
        _bump("<2 observed classes overall", len(files))
        stats["mv_fallback"] = len(files)
        _WS_STATS_BY_ATTR[attribute] = stats
        return ["" for _ in files], False, []

    try:
        from snorkel.labeling.model import LabelModel
        lm = LabelModel(cardinality=WS_NUM_ELTS, verbose=False)
        fit_kwargs = {"n_epochs": 100, "seed": 123}
        if y_dev:  # N3: class balance from sampled-gold indices (add-1 smoothed → no zero-prob class)
            cb = np.bincount(y_dev, minlength=WS_NUM_ELTS).astype(float) + 1.0
            fit_kwargs["class_balance"] = cb / cb.sum()
        stats["prior"] = "sampled_gold" if y_dev else "uniform"
        lm.fit(L, **fit_kwargs)
        preds_int = lm.predict(L, tie_break_policy="abstain")
    except Exception as e:  # noqa: BLE001
        _bump(f"snorkel error: {type(e).__name__}", len(files))
        stats["mv_fallback"] = len(files)
        stats["ws_error"] = f"{type(e).__name__}: {e}"
        _WS_STATS_BY_ATTR[attribute] = stats
        return ["" for _ in files], False, []

    mapped: list[str] = []
    for i in range(len(files)):
        ci = int(preds_int[i])
        name = int_to_name[i].get(ci, "")
        if ci < 0 or not name or name.startswith("dummy"):
            mapped.append("")  # combine_extractions will MV this doc
            stats["mv_fallback"] += 1
            _bump("doc predict abstain/dummy/tie")
        else:
            mapped.append(name)
            stats["ws_applied"] += 1
    _WS_STATS_BY_ATTR[attribute] = stats
    return mapped, False, []


# ── chunking (upstream txt parser, in-memory — no temp files) ────────────────

def _build_file2chunks(up, docs: dict, chunk_size: int) -> tuple[dict, dict]:
    """Build upstream-style file2chunks/file2contents from {name: text}.

    Uses upstream `get_txt_parse` so chunking is identical to upstream; doc names
    are the dict keys (upstream's synthesis/scoring/apply read from these dicts,
    never from disk).
    """
    file2contents = {name: (text or "") for name, text in docs.items()}
    file2chunks = {}
    for name, text in file2contents.items():
        _, chunks = up["get_txt_parse"](text, chunk_size=chunk_size, mode="train")
        file2chunks[name] = chunks
    return file2chunks, file2contents


# ── variant: direct (upstream get_model_extractions) ─────────────────────────

def _run_direct(be, up, *, attribute, all_docs, chunk_size, max_extract_chunks):
    """Per-doc direct LLM extraction via upstream get_model_extractions.

    Runs per doc so the apply (extraction-phase) token spend can be attributed
    per doc (the orchestrator uses it as retrieved_token_count for Direct).
    """
    be.set_phase("extraction")
    file2chunks, file2contents = _build_file2chunks(up, all_docs, chunk_size)
    file2chunks = up["filter_file2chunks"](file2chunks, list(all_docs.keys()), attribute) or {}

    dummy_session = [{"__name": "llm_backend"}]
    preds: dict[str, dict] = {}
    for doc_name in all_docs:
        chunks = file2chunks.get(doc_name, [])
        if max_extract_chunks and max_extract_chunks > 0:
            chunks = chunks[:max_extract_chunks]
        tok_before = sum(int(r.get("total_tokens", 0) or 0) for r in be.ledger)
        results, _toks, _err = up["get_model_extractions"](
            {doc_name: chunks}, [doc_name], attribute, dummy_session, "fm",
            collecting_preds=True,
        )
        tok_after = sum(int(r.get("total_tokens", 0) or 0) for r in be.ledger)
        # results[doc_name] is a list of [span,...] lists; flatten + clean + dedup.
        spans, seen = [], set()
        for ext in results.get(doc_name, []):
            for piece in (ext if isinstance(ext, list) else [ext]):
                cleaned = up["clean_function_predictions"](piece, attribute=attribute)
                if cleaned and cleaned.lower() not in ("none", "[]", "") and cleaned not in seen:
                    seen.add(cleaned)
                    spans.append(cleaned)
        preds[doc_name] = {"predicted": ", ".join(spans), "extract_llm_tokens": tok_after - tok_before}
    return preds


# ── variant: code / codeplus (upstream synthesis → score → select → combine) ──

def _run_code(be, up, *, attribute, sampled_docs, all_docs, chunk_size,
              topk, codeplus, combiner, extraction_fraction_thresh, sampled_gold_for_q=None):
    """Drive the upstream ClosedIE function pipeline for one attribute.

    Mirrors the algorithmic core of upstream `run_profiler` (do_end_to_end=False):
      filter_file2chunks → get_all_extractions → evaluate → get_topk_scripts_per_field
      → apply_final_ensemble (all docs) → combine_extractions (MV or N2 snorkel WS).
    All LLM work here is function synthesis/selection → phase="synthesis".
    Returns (preds, functions_meta, ws_stats|None).
    """
    be.set_phase("synthesis")
    file2chunks, file2contents = _build_file2chunks(up, all_docs, chunk_size)
    sample_files = list(sampled_docs.keys())
    all_files = list(all_docs.keys())

    file2chunks = up["filter_file2chunks"](file2chunks, sample_files, attribute)
    if file2chunks is None:
        # No keyword chunks for this attribute anywhere in the sample → no functions.
        return ({dn: {"predicted": "", "extract_llm_tokens": 0} for dn in all_docs}, [], None)

    # GOLD_KEY / "fm" are opaque dict keys + a non-"flan" model_name (selects upstream's
    # CONTEXT prompts); the ACTUAL model is the single llm_backend (see `model` above).
    GOLD_KEY = "gold_extraction"
    EXTRACTION_MODELS = ["fm"]
    manifest_sessions = {GOLD_KEY: [{"__name": "llm_backend"}],
                         "fm": [{"__name": "llm_backend"}]}
    args = SimpleNamespace(gold_extractions_file="__unused_uniform_prior__",
                           data_lake="evaporate")
    combiner_mode = "ws" if (codeplus and combiner == "ws") else "mv"
    num_top_k = 1 if not codeplus else max(1, topk)

    # PREDICT: noisy GOLD_KEY extraction on sampled + synthesize functions + apply to sampled.
    all_extractions, function_dictionary, _ = up["get_all_extractions"](
        file2chunks, file2contents, sample_files, attribute, manifest_sessions,
        EXTRACTION_MODELS, GOLD_KEY, args, use_qa_model=False, overwrite_cache=False,
    )
    if not all_extractions or not isinstance(function_dictionary, dict) or not function_dictionary:
        return ({dn: {"predicted": "", "extract_llm_tokens": 0} for dn in all_docs}, [], None)

    # SCORE: upstream noisy-LLM-gold F1 (with e/τ abstention) — function selection only.
    all_metrics, _key2golds, _ = up["evaluate"](
        all_extractions, GOLD_KEY, field=attribute,
        manifest_session=manifest_sessions[GOLD_KEY], overwrite_cache=False,
        combiner_mode=combiner_mode, extraction_fraction_thresh=extraction_fraction_thresh,
        use_abstension=True,
    )
    selected_keys = up["get_topk_scripts_per_field"](
        all_metrics, function_dictionary, all_extractions, gold_key=GOLD_KEY,
        k=num_top_k, do_end_to_end=False, combiner_mode=combiner_mode,
    )

    functions_meta = [
        {"source": (function_dictionary.get(k, {}) or {}).get("function"),
         "score": (all_metrics.get(k, {}) or {}).get("average_f1", 0.0),
         "from_doc": None}
        for k in function_dictionary
    ]

    if not selected_keys:
        return ({dn: {"predicted": "", "extract_llm_tokens": 0} for dn in all_docs},
                functions_meta, None)

    # APPLY: run selected functions on ALL docs (pure Python, sandboxed by upstream).
    top_k_extractions, _ = up["apply_final_ensemble"](
        all_files, file2chunks, file2contents, selected_keys, all_metrics, attribute,
        function_dictionary, data_lake="evaporate", function_cache=False,
        manifest_sessions=manifest_sessions, MODELS=EXTRACTION_MODELS,
        overwrite_cache=False, do_end_to_end=False,
    )

    # COMBINE: MV, or N2 snorkel WS via the patched profiler.run_ws.
    _WS_STATS_BY_ATTR.pop(attribute, None)
    # N3: hand the SAMPLED docs' gold (held-in labels) to the WS class prior; unsampled
    # gold is excluded (apply-time leak). Read inside the run_ws shim.
    _SAMPLED_GOLD_BY_ATTR[attribute] = {
        dn: g for dn, g in (sampled_gold_for_q or {}).items() if g is not None
    }
    file2metadata, _ = up["combine_extractions"](
        args, top_k_extractions, all_metrics, combiner_mode=combiner_mode,
        train_extractions=all_extractions, attribute=attribute, gold_key=GOLD_KEY,
        extraction_fraction_thresh=extraction_fraction_thresh,
    )

    preds = {dn: {"predicted": str(file2metadata.get(dn, "") or ""), "extract_llm_tokens": 0}
             for dn in all_docs}
    ws_stats = None
    if codeplus:
        ws_stats = _WS_STATS_BY_ATTR.get(attribute) or {
            "combiner": combiner, "n_docs": len(all_files),
            "ws_applied": 0, "mv_fallback": len(all_files),
            "fallback_reasons": {"combiner=mv": len(all_files)} if combiner == "mv" else {},
        }
    return preds, functions_meta, ws_stats


# ── main ──────────────────────────────────────────────────────────────────

def run(input_path: Path, output_path: Path) -> None:
    global _BACKEND

    spec = json.loads(input_path.read_text(encoding="utf-8"))
    missing = [k for k in _REQUIRED_INPUT_KEYS if k not in spec]
    if missing:
        raise ValueError(f"input missing keys: {missing}")
    if spec["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {spec['schema_version']}")

    variant = spec["variant"]
    if variant not in ("direct", "code", "codeplus"):
        raise ValueError(f"unknown variant {variant!r}")

    up = _import_upstream()
    from llm_backend import LLMBackend  # noqa: E402  (same dir, on path)
    cfg = spec.get("config", {})
    model = str(cfg.get("model", "gpt54"))  # match the LSF variant being compared
    _BACKEND = LLMBackend(model=model)
    be = _BACKEND

    chunk_size = int(cfg.get("chunk_chars", 3000))  # upstream configs.py default
    topk = int(cfg.get("topk", 10))                 # upstream num_top_k_scripts default
    max_extract_chunks = int(cfg.get("max_extract_chunks", 40))
    combiner = str(cfg.get("combiner", "ws"))
    extraction_fraction_thresh = float(cfg.get("extraction_fraction_thresh", 0.9))

    sampled_docs = spec["sampled_docs"]
    all_docs = spec["all_docs"]
    sampled_gold = spec.get("sampled_gold", {})  # {q_slug: {sampled_doc: gold}} — N3 prior

    predictions: dict[str, dict] = {}
    functions: dict[str, list] = {}
    codeplus_ws: dict[str, dict] = {}
    errors: list[str] = []

    for q in spec["questions"]:
        q_slug = q["slug"]
        attribute = q["text"].lower()  # mirror upstream run_profiler:602 (case-normalize before prompts)
        try:
            if variant == "direct":
                preds = _run_direct(
                    be, up, attribute=attribute, all_docs=all_docs,
                    chunk_size=chunk_size, max_extract_chunks=max_extract_chunks,
                )
            else:
                preds, fns, ws_stats = _run_code(
                    be, up, attribute=attribute, sampled_docs=sampled_docs,
                    all_docs=all_docs, chunk_size=chunk_size, topk=topk,
                    codeplus=(variant == "codeplus"), combiner=combiner,
                    extraction_fraction_thresh=extraction_fraction_thresh,
                    sampled_gold_for_q=sampled_gold.get(q_slug, {}),
                )
                functions[q_slug] = fns
                if ws_stats is not None:
                    codeplus_ws[q_slug] = ws_stats
            predictions[q_slug] = preds
        except Exception as e:  # noqa: BLE001
            errors.append(f"{q_slug}: {type(e).__name__}: {e}")
            predictions[q_slug] = {dn: {"predicted": "", "extract_llm_tokens": 0} for dn in all_docs}

    payload = {
        "schema_version": SCHEMA_VERSION,
        "variant": variant,
        "combiner": combiner if variant == "codeplus" else None,
        "predictions": predictions,
        "functions": functions,
        "codeplus_ws": codeplus_ws,
        "ledger": be.ledger,
        "ledger_summary": be.ledger_summary(),
        "errors": errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[run_variant:{variant}] wrote {output_path} "
          f"(questions={len(predictions)}, errors={len(errors)})", flush=True)


def _self_test() -> int:
    """A4 gate: prove upstream imports + the N1/N2 patches install, no network.

    Imports upstream, asserts `evaporate.utils.get_response` is our patch and
    `evaporate.profiler.run_ws` is the snorkel shim, then exercises the snorkel WS
    shim on a synthetic vote set (no LLM, no gold file).
    """
    up = _import_upstream()  # noqa: F841
    import evaporate.utils as up_utils
    import evaporate.profiler as up_profiler
    assert up_utils.get_response is _patched_get_response, "N1 patch not installed"
    assert up_profiler.run_ws is _snorkel_run_ws, "N2 shim not installed"

    # WS shim parity smoke: 3 docs, 3 functions, clear majority per doc.
    votes = {
        "d1": ["$550", "$550", "$550"],
        "d2": ["k143467", "k143467", ""],
        "d3": ["$550", "$550", "$550"],
    }
    preds, used_deps, missing = _snorkel_run_ws(votes, "__none__", attribute="price")
    assert used_deps is False and missing == [], (used_deps, missing)
    assert len(preds) == 3, preds
    assert "price" in _WS_STATS_BY_ATTR, _WS_STATS_BY_ATTR
    print("[run_variant:self-test] OK — N1/N2 patches installed; WS shim ran:",
          {"preds": preds, "ws_stats": _WS_STATS_BY_ATTR["price"]})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaporate variant runner (in-venv subprocess target)")
    ap.add_argument("--input", type=Path, help="staged input JSON")
    ap.add_argument("--output", type=Path, help="output JSON path")
    ap.add_argument("--self-test", action="store_true", help="A4 gate: import upstream + install patches, no LLM")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    if not args.input or not args.output:
        ap.error("--input and --output are required unless --self-test")
    run(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
