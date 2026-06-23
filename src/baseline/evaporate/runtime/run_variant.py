"""Evaporate variant runner — subprocess target run INSIDE `.venv-evaporate`.

Thin interception over upstream: drives the real upstream Evaporate code
(chunking, synthesis, scoring, selection, sandboxed exec, combine) and deviates
in only four seams:
  N1  LLM backend — `evaporate.utils.get_response` patched to `llm_backend`.
  N2  Code+ WS    — snorkel 0.10 `LabelModel` shim replacing the dead MeTaL `run_ws`.
  N3  WS prior    — class balance from the SAMPLED docs' gold only (no apply-set leak).
  N4  scoring/split/cost — owned by `run_evaporate.py`, reusing pipeline.py.

Variants: direct (per-doc LLM extract), code (best-1 synthesized fn), codeplus
(top-k fns + WS/MV combine). I/O: input keys in `_REQUIRED_INPUT_KEYS`; output is
per-q/per-doc predictions + functions + codeplus_ws stats + a phase-tagged ledger.
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

WS_NUM_ELTS = 5  # LabelModel cardinality = upstream get_data num_elts default
_MIN_COMPLETION_TOKENS = 256  # N1: floor output budget (reasoning models eat tiny max_toks)

_HERE = Path(__file__).resolve().parent
_UPSTREAM = _HERE.parent / "upstream"

# Module globals: the WS shim has a fixed upstream signature, so it reads its inputs
# (cleaner, sampled gold, stats) from here rather than as arguments.
_BACKEND: Any = None
_CLEAN_COMPARISON: Any = None
_WS_STATS_BY_ATTR: dict[str, dict] = {}
_SAMPLED_GOLD_BY_ATTR: dict[str, dict] = {}  # {attr: {sampled_doc: gold}} — N3 prior


# ── N1: LLM backend interception ─────────────────────────────────────────────

def _patched_get_response(prompt, manifest, overwrite=False, max_toks=10,
                          stop_token=None, gold_choices=None, verbose=False):
    """N1 drop-in for `evaporate.utils.get_response` → `llm_backend`.

    Patching the utils-module global reaches every upstream call (they all go via
    `apply_prompt`, which resolves `get_response` at call time). No `stop` sent.
    """
    # gold_choices (upstream's constrained branch) is unused in ClosedIE — fail loud.
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
    """Stub upstream's top-level `manifest` and metal `run_ws` imports (we replace
    both — N1 backend, N2 snorkel shim) so importing the profiler succeeds."""
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

    _CLEAN_COMPARISON = clean_comparison  # the exact cleaner upstream get_data uses

    up_utils.get_response = _patched_get_response   # N1
    up_profiler.run_ws = _snorkel_run_ws            # N2 (combine_extractions calls it here)

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
    """N2/N3 drop-in for upstream `run_ws`. Same `(mapped_preds, used_deps,
    missing_files)` contract; `mapped_preds` aligns to `all_votes` order and
    `combine_extractions` handles MV fallbacks.

    Label space mirrors upstream `get_data` (top-WS_NUM_ELTS rank-based classes,
    abstain bucket, dummy padding, seed-0 shuffle, -1 abstain). N3: the class prior
    is from the SAMPLED docs' gold only (via `_SAMPLED_GOLD_BY_ATTR`), never the
    apply-set gold (the leak). cvxpy dep-learning dropped (= upstream's no-deps path).
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
    """Build file2chunks/file2contents from {name: text} via upstream `get_txt_parse`
    (same chunking as upstream; upstream reads from these dicts, not disk)."""
    file2contents = {name: (text or "") for name, text in docs.items()}
    file2chunks = {}
    for name, text in file2contents.items():
        _, chunks = up["get_txt_parse"](text, chunk_size=chunk_size, mode="train")
        file2chunks[name] = chunks
    return file2chunks, file2contents


# ── variant: direct (upstream get_model_extractions) ─────────────────────────

def _run_direct(be, up, *, attribute, all_docs, chunk_size, max_extract_chunks):
    """Per-doc direct LLM extraction via upstream `get_model_extractions` (per doc
    so the extraction-phase token spend is attributable per doc)."""
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
    """Upstream ClosedIE pipeline for one attribute (= run_profiler do_end_to_end=False):
    filter_file2chunks → get_all_extractions → evaluate → get_topk → apply_final_ensemble
    → combine_extractions. All LLM work is synthesis-phase."""
    be.set_phase("synthesis")
    file2chunks, file2contents = _build_file2chunks(up, all_docs, chunk_size)
    sample_files = list(sampled_docs.keys())
    all_files = list(all_docs.keys())

    file2chunks = up["filter_file2chunks"](file2chunks, sample_files, attribute)
    if file2chunks is None:
        # No keyword chunks for this attribute anywhere in the sample → no functions.
        return ({dn: {"predicted": "", "extract_llm_tokens": 0} for dn in all_docs}, [], None)

    # GOLD_KEY / "fm" are opaque keys / a non-"flan" model_name; the real model is llm_backend.
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
    # N3: hand the WS prior the SAMPLED docs' gold only (read inside the shim).
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
    """A4 gate (no network): assert the N1/N2 patches installed, then run the WS shim
    on a synthetic vote set."""
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
