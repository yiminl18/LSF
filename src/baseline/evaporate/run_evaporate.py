"""Evaporate baseline orchestrator — fair comparison vs LSF (llm-rule-gen).

Runs in the MAIN repo env; the LLM work happens in `.venv-evaporate` via
`runtime/run_variant.py`. The orchestrator makes Evaporate look like a pipeline.py
result by reusing pipeline.py's three seams: the same split (`stage_sampling`,
seed=0/cap=20 or the prebuilt financebench cluster), the same gpt54 judge
(`_JUDGE_SYSTEM`/`_judge`), and the same per-doc schema → `pipeline_summary.json`.
Cost is two-column (synthesis vs apply); `cost_ratio` = pipeline's
`retrieved_token_count / _count_tokens(doc)`.

Usage: python src/baseline/evaporate/run_evaporate.py --dataset court --variant code \\
       --model gpt54 --output-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pipeline  # noqa: E402  (the comparison target; we reuse its seams)

_RUNTIME = Path(__file__).resolve().parent / "runtime"
_HERE_UPSTREAM = Path(__file__).resolve().parent / "upstream"
_VENV_PY = _RUNTIME / ".venv-evaporate" / "bin" / "python"
_RUN_VARIANT = _RUNTIME / "run_variant.py"

VARIANTS = ("direct", "code", "codeplus")

# Max chars of `predicted` fed to the judge (~2k tokens). Guards the judge model's
# 32k context against degenerate huge extractions; valid answers are far shorter.
_JUDGE_PRED_CHAR_CAP = 8000

def _judge_usage_totals(judge_mod) -> dict:
    """Snapshot the judge client's cumulative usage from the per-call recorder
    (models.<judge>.LLM_DB.totals). Cache-aware: a cache hit returns its recorded
    tokens and is counted too, so a delta over the scoring loop = this run's judge
    tokens regardless of cache state. Returns zeros if the recorder isn't present."""
    db = getattr(judge_mod, "LLM_DB", None)
    t = getattr(db, "totals", None) or {}
    return {"calls": int(t.get("calls", 0)),
            "input_tokens": int(t.get("input_tokens", 0)),
            "output_tokens": int(t.get("output_tokens", 0))}


# ── doc / query plumbing (mirrors pipeline.py conventions) ──────────────────

def _doc_text(doc: dict) -> str:
    """Join a reconstructed-JSON doc's text spans — the SAME string pipeline.py
    uses as the cost denominator base (`pipeline.stage_apply_and_eval`)."""
    return "\n".join(s.get("text", "") for s in doc.get("texts", []))


# Default queries per dataset (mirrors pipeline.py): court/nopv → queries.json;
# financebench → sample_queries.txt (pipeline.py default; no queries.json there).
_DEFAULT_QUERIES = {
    "financebench": "data/financebench/sample_queries.txt",
    "court": "data/court/queries.json",
    "nopv": "data/nopv/queries.json",
}


def _resolve_queries(dataset: str, queries_file: str | None) -> tuple[list[str], str]:
    """Return (questions, resolved_source_path) so the run records exactly which
    query file produced the question set (provenance for review)."""
    if queries_file:
        return pipeline._load_queries(queries_file), queries_file
    rel = _DEFAULT_QUERIES.get(dataset)
    cand = (_ROOT / rel) if rel else (_ROOT / "data" / dataset / "queries.json")
    if cand.exists():
        return pipeline._load_queries(str(cand)), str(cand)
    raise FileNotFoundError(
        f"no --queries given and the default for dataset={dataset!r} is missing: {cand}. "
        f"Pass --queries explicitly (never silently fall back to a different question set)."
    )


# ── staging + subprocess ────────────────────────────────────────────────────

def _stage_input(
    *, variant: str, questions: list[dict], sample_docs: dict, all_docs: dict,
    sampled_gold: dict, cfg: dict, staging_dir: Path,
) -> Path:
    staging_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "schema_version": 1,
        "variant": variant,
        "dataset": cfg.get("dataset"),
        "questions": questions,
        "sampled_docs": sample_docs,
        "all_docs": all_docs,
        "sampled_gold": sampled_gold,
        "config": cfg.get("variant_config", {}),
    }
    p = staging_dir / "input.json"
    p.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    return p


def _run_variant_subprocess(input_path: Path, output_path: Path, timeout: int,
                            log_path: Path | None = None) -> dict:
    if not _VENV_PY.exists():
        raise FileNotFoundError(
            f"isolated venv python not found at {_VENV_PY}. "
            f"Run: bash {_RUNTIME / 'setup_env.sh'}"
        )
    cmd = [str(_VENV_PY), str(_RUN_VARIANT), "--input", str(input_path), "--output", str(output_path)]
    print(f"  [evaporate] subprocess: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True, timeout=timeout)
    # B: persist the FULL subprocess stdout/stderr on every run (not just on failure)
    # — upstream's synthesis/scoring progress is the run's only execution log.
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"$ {' '.join(cmd)}\n(rc={proc.returncode})\n\n"
            f"=== STDOUT ===\n{proc.stdout}\n\n=== STDERR ===\n{proc.stderr}\n",
            encoding="utf-8",
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"run_variant failed (rc={proc.returncode}).\n"
            + (f"Full log: {log_path}\n" if log_path else "")
            + f"STDOUT tail:\n{proc.stdout[-1500:]}\nSTDERR tail:\n{proc.stderr[-1500:]}"
        )
    if not output_path.exists():
        raise RuntimeError(f"run_variant returned 0 but no output at {output_path}")
    return json.loads(output_path.read_text(encoding="utf-8"))


def _merge_var_outs(outs: list[dict]) -> dict:
    """Merge per-question run_variant outputs (question-level parallelism) into the
    single var_out shape the scorer expects. Per-question dicts are disjoint by slug;
    the ledger is concatenated and `ledger_summary.by_phase` recomputed."""
    base = next((o for o in outs if o), {})
    merged: dict[str, Any] = {
        "schema_version": base.get("schema_version"),
        "variant": base.get("variant"),
        "combiner": base.get("combiner"),
        "predictions": {}, "functions": {}, "codeplus_ws": {}, "selection": {},
        "apply_timing": {},
        "ledger": [], "errors": [],
    }
    for o in outs:
        if not o:
            continue
        for k in ("predictions", "functions", "codeplus_ws", "selection", "apply_timing"):
            merged[k].update(o.get(k, {}) or {})
        merged["ledger"].extend(o.get("ledger", []) or [])
        merged["errors"].extend(o.get("errors", []) or [])

    agg = {p: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
           for p in ("synthesis", "extraction")}
    for r in merged["ledger"]:
        b = agg.get(r.get("phase"))
        if b:
            b["calls"] += 1
            for t in ("prompt_tokens", "completion_tokens", "total_tokens"):
                b[t] += int(r.get(t, 0) or 0)
    base_ls = base.get("ledger_summary", {}) or {}
    merged["ledger_summary"] = {
        "model": base_ls.get("model"), "provider": base_ls.get("provider"),
        "deployment": base_ls.get("deployment"), "by_phase": agg,
        "n_calls": len(merged["ledger"]), "credential_source": base_ls.get("credential_source"),
    }
    return merged


# ── scoring (reuses pipeline's judge + cost denominator) ─────────────────────

def _score_split(
    *, variant: str, split: str, question: str, question_slug: str,
    doc_map: dict, labels: dict, var_preds: dict, judge_mod, apply_root: Path,
    rule_set_slug: str, judge_workers: int = 8,
) -> dict:
    """Build pipeline-schema per-doc records + aggregate for one (question, split).

    Judge calls are independent per doc, so they run concurrently (thread pool —
    the OpenAI client is thread-safe); `executor.map` preserves doc order. Set
    `judge_workers<=1` to force the sequential path.
    """
    run_dir = apply_root / question_slug / split
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_records: list[dict] = []
    per_doc: list[dict] = []

    # Pass 1 (no network): assemble per-doc inputs + cost denominators.
    items: list[tuple] = []
    for doc_name, doc in doc_map.items():
        pred_rec = var_preds.get(doc_name, {})
        predicted = pred_rec.get("predicted", "")
        gt = labels.get(doc_name + ".pdf", {}).get(question)
        if gt is None:
            gt = labels.get(doc_name, {}).get(question)

        total_tok = pipeline._count_tokens(_doc_text(doc))
        if variant == "direct":
            ret_tok = int(pred_rec.get("extract_llm_tokens", 0) or 0)
        else:
            ret_tok = pipeline._count_tokens(str(predicted)) if predicted else 0
        items.append((doc_name, predicted, gt, total_tok, ret_tok))

    # Pass 2 (network): judge each doc, concurrently. Order preserved by map().
    # Cap the predicted text fed to the judge: a degenerate synthesized function can
    # dump a huge span (seen ~64k tokens on financebench), blowing the judge model's
    # 32k context (400 BadRequest). A valid answer is short, so truncating only ever
    # affects garbage extractions (still judged INCORRECT). Any judge error → False,
    # so one bad doc can't crash the whole dataset. (Storage/cost use the FULL string.)
    def _judge_one(it: tuple) -> bool:
        _dn, predicted, gt, _tt, _rt = it
        pred_for_judge = predicted[:_JUDGE_PRED_CHAR_CAP] if isinstance(predicted, str) else predicted
        try:
            return pipeline._judge(judge_mod, question, gt, pred_for_judge)
        except Exception as e:  # noqa: BLE001
            print(f"  [judge:WARN] {_dn} judged INCORRECT after error: {type(e).__name__}", flush=True)
            return False

    if judge_workers and judge_workers > 1 and len(items) > 1:
        with ThreadPoolExecutor(max_workers=judge_workers) as ex:
            verdicts = list(ex.map(_judge_one, items))
    else:
        verdicts = [_judge_one(it) for it in items]

    for (doc_name, predicted, gt, total_tok, ret_tok), correct in zip(items, verdicts):
        raw_records.append({
            "doc_name": doc_name,
            "predicted_answer": predicted,
            "retrieved_token_count": ret_tok,
            "latency_seconds": 0.0,
            "input_tokens": 0,
            "used_fallback": False,
        })
        per_doc.append({
            "doc_name": doc_name,
            "predicted": predicted,
            "ground_truth": gt,
            "correct": correct,
            "latency_seconds": 0.0,
            "retrieved_tokens": ret_tok,
            "input_tokens": 0,
            "cost_ratio": (ret_tok / total_tok) if total_tok > 0 else 0.0,
            "used_fallback": False,
        })

    pipeline._write_json(run_dir / f"{rule_set_slug}.json", raw_records)

    eval_record = {
        "question": question,
        "question_slug": question_slug,
        "split": split,
        "apply_strategy": f"evaporate_{variant}",
        "n": len(per_doc),
        "n_correct": sum(1 for r in per_doc if r["correct"]),
        "accuracy": round(mean(int(r["correct"]) for r in per_doc), 4) if per_doc else 0.0,
        "avg_latency": 0.0,
        "avg_retrieved": round(mean(r["retrieved_tokens"] for r in per_doc), 1) if per_doc else 0.0,
        "avg_input_tok": 0.0,
        "avg_cost_ratio": round(mean(r["cost_ratio"] for r in per_doc), 6) if per_doc else 0.0,
        "per_doc": per_doc,
    }
    pipeline._write_json(apply_root / f"{question_slug}_{split}.json", eval_record)
    print(f"  [eval:{split}] {question_slug}  acc={eval_record['accuracy']:.3f} "
          f"cost_ratio={eval_record['avg_cost_ratio']:.4f}", flush=True)
    return {k: v for k, v in eval_record.items() if k != "per_doc"}


# ── orchestration ────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> dict:
    apply_root = Path(args.output_dir)
    apply_root.mkdir(parents=True, exist_ok=True)
    rule_set_slug = f"evaporate_{args.variant}"
    start_dt = datetime.now(timezone.utc)
    t0 = time.time()

    processing_dir = args.processing_dir or str(pipeline._default_processing_dir(args.dataset))
    queries, queries_src = _resolve_queries(args.dataset, args.queries)
    if args.limit_questions:
        queries = queries[: args.limit_questions]
    questions = [{"slug": pipeline._make_slug(q), "text": q} for q in queries]

    # ── Seam 1: SAME split as pipeline.py ──
    # FinanceBench uses a PREBUILT cluster split; guard a wrong/missing --cluster
    # (note 'multi_cluster' is SINGULAR), which would silently fall back to a fresh
    # random split ≠ the LSF run. Fail loudly instead.
    if args.dataset == "financebench" and args.sampling_strategy == "random":
        fb_split = _ROOT / "data" / "financebench" / "sample" / args.cluster / "random"
        if not (fb_split / "sample_doc_labels.json").exists():
            available = sorted(p.name for p in (_ROOT / "data/financebench/sample").glob("*") if p.is_dir())
            raise FileNotFoundError(
                f"financebench prebuilt split not found: {fb_split}/sample_doc_labels.json. "
                f"Pass --cluster matching a real split dir (available: {available}; "
                f"note 'multi_cluster' is SINGULAR)."
            )
    sample_labels_path, unsampled_labels_path = pipeline.stage_sampling(
        strategy=args.sampling_strategy, dataset=args.dataset, cluster=args.cluster,
        output_dir=apply_root, skip_existing=args.skip_existing, processing_dir=processing_dir,
    )
    sample_labels = json.loads(Path(sample_labels_path).read_text(encoding="utf-8"))
    unsampled_labels = json.loads(Path(unsampled_labels_path).read_text(encoding="utf-8"))
    print(f"  [split] sampled={len(sample_labels)} unsampled={len(unsampled_labels)} "
          f"(strategy={args.sampling_strategy}, src={sample_labels_path})", flush=True)

    sample_docs_full = pipeline._load_docs(sample_labels, processing_dir)
    unsampled_docs_full = pipeline._load_docs(unsampled_labels, processing_dir)
    if args.limit_sampled:  # smoke only: shrinks synthesis set, breaks the LSF-parity split
        keep = list(sample_docs_full)[: args.limit_sampled]
        sample_docs_full = {k: sample_docs_full[k] for k in keep}
    if args.limit_unsampled:  # smoke control only
        keep = list(unsampled_docs_full)[: args.limit_unsampled]
        unsampled_docs_full = {k: unsampled_docs_full[k] for k in keep}

    sample_text = {dn: _doc_text(d) for dn, d in sample_docs_full.items()}
    all_docs_full = {**sample_docs_full, **unsampled_docs_full}
    all_text = {dn: _doc_text(d) for dn, d in all_docs_full.items()}

    # ── A: launch-time provenance (rewritten with timing + counts at completion;
    # if the run crashes mid-way this 'running' record + the subprocess log remain) ──
    _meta_kw = dict(
        start_dt=start_dt, queries=queries, queries_src=queries_src,
        processing_dir=processing_dir, sample_path=sample_labels_path,
        unsampled_path=unsampled_labels_path, n_sampled=len(sample_docs_full),
        n_unsampled=len(unsampled_docs_full),
    )
    _write_run_metadata(apply_root, args, status="running", **_meta_kw)

    sampled_gold = {
        q["slug"]: {
            dn: (sample_labels.get(dn + ".pdf", {}) or sample_labels.get(dn, {})).get(q["text"])
            for dn in sample_docs_full
        }
        for q in questions
    }

    # ── stage + run the variant in the isolated venv ──
    staging_dir = apply_root / "staging" / args.variant
    cfg = {
        "dataset": args.dataset,
        "variant_config": {
            "model": args.model,
            "chunk_chars": args.chunk_chars,
            "num_functions": args.num_functions,
            "topk": args.topk,
            "max_extract_chunks": args.max_extract_chunks,
            "combiner": args.combiner,
        },
    }
    output_path = staging_dir / "output.json"
    if args.question_workers and args.question_workers > 1 and len(questions) > 1:
        # Question-level parallelism: questions are independent, but upstream
        # synthesis is sequential WITHIN a run_variant subprocess. Shard one
        # question per subprocess and run up to `question_workers` concurrently
        # (each fires ~1 LLM call at a time → far under Pioneer's 1200/min), then
        # merge. This is the big speedup — synthesis is the serial bottleneck.
        def _run_shard(iq: tuple[int, dict]) -> dict:
            i, q = iq
            sdir = staging_dir / f"q{i:03d}"
            ip = _stage_input(
                variant=args.variant, questions=[q], sample_docs=sample_text,
                all_docs=all_text, sampled_gold={q["slug"]: sampled_gold.get(q["slug"], {})},
                cfg=cfg, staging_dir=sdir,
            )
            lp = apply_root / "logs" / f"run_variant.{args.variant}.q{i:03d}.log"
            return _run_variant_subprocess(ip, sdir / "output.json", timeout=args.timeout, log_path=lp)

        print(f"  [parallel] {len(questions)} questions across {args.question_workers} workers", flush=True)
        with ThreadPoolExecutor(max_workers=args.question_workers) as ex:
            shard_outs = list(ex.map(_run_shard, list(enumerate(questions))))
        var_out = _merge_var_outs(shard_outs)
        pipeline._write_json(output_path, var_out)  # merged, for the artifacts index
    else:
        input_path = _stage_input(
            variant=args.variant, questions=questions, sample_docs=sample_text,
            all_docs=all_text, sampled_gold=sampled_gold, cfg=cfg, staging_dir=staging_dir,
        )
        log_path = apply_root / "logs" / f"run_variant.{args.variant}.log"
        var_out = _run_variant_subprocess(input_path, output_path, timeout=args.timeout, log_path=log_path)

    # ── completeness guard (acceptance #4) ──
    expected_docs = set(all_text)
    shortfall = {}
    for q in questions:
        got = set((var_out.get("predictions", {}).get(q["slug"], {})).keys())
        missing = expected_docs - got
        if missing:
            shortfall[q["slug"]] = sorted(missing)
    if shortfall:
        fail = {"status": "FAILED", "reason": "incomplete predictions", "missing": shortfall}
        pipeline._write_json(apply_root / "evaporate_FAILED.json", fail)
        raise RuntimeError(f"completeness guard: missing predictions for {len(shortfall)} question(s); "
                           f"see {apply_root / 'evaporate_FAILED.json'}")

    # ── variant-error guard ──
    # run_variant fills a failed question's docs with empty preds (which would pass
    # the completeness guard and silently score 0%). Fail the run loudly instead.
    var_errors = var_out.get("errors", []) or []
    if var_errors:
        fail = {"status": "FAILED", "reason": "variant subprocess reported per-question errors",
                "errors": var_errors}
        pipeline._write_json(apply_root / "evaporate_FAILED.json", fail)
        raise RuntimeError(
            f"variant-error guard: run_variant reported {len(var_errors)} question error(s) "
            f"(synthesis/extraction failed → would falsely score 0%); see "
            f"{apply_root / 'evaporate_FAILED.json'}:\n  " + "\n  ".join(var_errors)
        )

    # ── Seam 2+3: score with pipeline's judge, emit pipeline schema ──
    # Per-call judge usage log: models.gpt54 installs azure_local.install_usage_logging
    # at import IF LSF_LLM_USAGE_LOG is set, so every judge chat.completions.create
    # appends one {model,prompt_tokens,completion_tokens}. Set it BEFORE the import.
    # --no-judge: skip the entire scoring phase (no accuracy) — used for latency-only
    # runs where judge LLM calls would otherwise compete with synthesis for the
    # provider rate limit (and 429-backoff would corrupt the synthesis latency).
    run_judge = not getattr(args, "no_judge", False)
    judge_mod = None
    if run_judge:
        judge_usage_path = apply_root / "judge_usage.jsonl"
        os.environ["LSF_LLM_USAGE_LOG"] = str(judge_usage_path)
        judge_mod = __import__("importlib").import_module(f"models.{args.judge_model}")
        assert pipeline._JUDGE_SYSTEM, "pipeline judge system prompt missing"
    # Snapshot the judge client's cumulative usage so the delta over the scoring
    # loop = this run's judge tokens (cache-aware: cache hits return recorded
    # tokens and still count). No per-run db query / run_id needed.
    _jt_before = _judge_usage_totals(judge_mod)

    per_question_summary: list[dict] = []
    for q in questions:
        q_slug, question = q["slug"], q["text"]
        preds = var_out.get("predictions", {}).get(q_slug, {})
        if run_judge:
            sampled_eval = _score_split(
                variant=args.variant, split="sampled", question=question, question_slug=q_slug,
                doc_map=sample_docs_full, labels=sample_labels,
                var_preds={dn: preds.get(dn, {}) for dn in sample_docs_full},
                judge_mod=judge_mod, apply_root=apply_root, rule_set_slug=rule_set_slug,
                judge_workers=args.judge_workers,
            )
            unsampled_eval = _score_split(
                variant=args.variant, split="unsampled", question=question, question_slug=q_slug,
                doc_map=unsampled_docs_full, labels=unsampled_labels,
                var_preds={dn: preds.get(dn, {}) for dn in unsampled_docs_full},
                judge_mod=judge_mod, apply_root=apply_root, rule_set_slug=rule_set_slug,
                judge_workers=args.judge_workers,
            )
            # Combined acc = pool both splits' docs within the question, then this is
            # averaged across questions in `overall` (question-micro, dataset-macro —
            # same structure as sampled/unsampled above).
            comb_n = sampled_eval.get("n", 0) + unsampled_eval.get("n", 0)
            comb_correct = sampled_eval.get("n_correct", 0) + unsampled_eval.get("n_correct", 0)
            combined_accuracy = round(comb_correct / comb_n, 4) if comb_n else None
        else:
            sampled_eval = unsampled_eval = {}
            combined_accuracy = None
        sel = var_out.get("selection", {}).get(q_slug, {})
        at = var_out.get("apply_timing", {}).get(q_slug, {})  # online rule-application timing
        per_question_summary.append({
            "question": question,
            "question_slug": q_slug,
            "num_functions": len(var_out.get("functions", {}).get(q_slug, [])) or None,
            "n_selected": sel.get("n_selected"),
            "best_f1": sel.get("best_f1"),
            "threshold_bypass_fallback": sel.get("threshold_bypass_fallback"),
            "sampled_accuracy": sampled_eval.get("accuracy"),
            "unsampled_accuracy": unsampled_eval.get("accuracy"),
            "combined_accuracy": combined_accuracy,
            "avg_cost_ratio_sampled": sampled_eval.get("avg_cost_ratio"),
            "avg_cost_ratio_unsampled": unsampled_eval.get("avg_cost_ratio"),
            # ONLINE (Table-6 analog): per-doc pure-Python rule-application latency.
            "online_s_per_doc": at.get("online_s_per_doc"),
            "apply_seconds": at.get("apply_seconds"),
            "combine_seconds": at.get("combine_seconds"),
            "n_apply_docs": at.get("n_docs"),
            "llm_calls_in_window": at.get("llm_calls_in_window"),  # MUST be 0
        })

    sampled_accs = [q["sampled_accuracy"] for q in per_question_summary if q.get("sampled_accuracy") is not None]
    unsampled_accs = [q["unsampled_accuracy"] for q in per_question_summary if q.get("unsampled_accuracy") is not None]
    combined_accs = [q["combined_accuracy"] for q in per_question_summary if q.get("combined_accuracy") is not None]
    # Unsampled-only basis, matching pipeline.py's `overall.avg_cost_ratio`
    # (pipeline.py aggregates `avg_cost_ratio_unsampled`); same field name on purpose.
    cost_ratios = [q["avg_cost_ratio_unsampled"] for q in per_question_summary if q.get("avg_cost_ratio_unsampled") is not None]

    # ── Gap C: two-column cost from the phase-tagged ledger ──
    ledger_sum = var_out.get("ledger_summary", {})
    by_phase = ledger_sum.get("by_phase", {})
    synth = by_phase.get("synthesis", {})
    apply_extract = by_phase.get("extraction", {})

    # ── Measured cost in $ — both gen and judge from per-call token returns
    # (cache-aware: cached calls return their recorded tokens and are counted).
    # gen = evaporate backend ledger (synthesis+extraction); judge = the judge
    # client's usage delta over the scoring loop. Priced via src/llm_cost. ──
    import llm_cost as _llm_cost
    gen_in = int(synth.get("prompt_tokens", 0)) + int(apply_extract.get("prompt_tokens", 0))
    gen_out = int(synth.get("completion_tokens", 0)) + int(apply_extract.get("completion_tokens", 0))
    gen_provider = ledger_sum.get("provider") or "pioneer"
    gen_model = ledger_sum.get("deployment") or args.model
    _jt_after = _judge_usage_totals(judge_mod)
    jt_in = max(0, _jt_after["input_tokens"] - _jt_before["input_tokens"])
    jt_out = max(0, _jt_after["output_tokens"] - _jt_before["output_tokens"])
    jt_calls = max(0, _jt_after["calls"] - _jt_before["calls"])
    judge_provider = getattr(judge_mod, "PROVIDER", "pioneer")
    judge_model = getattr(judge_mod, "AZURE_DEPLOYMENT", "gpt-5.4")
    gen_cost = _llm_cost.compute_cost(gen_in, gen_out, gen_provider, gen_model)
    judge_cost = _llm_cost.compute_cost(jt_in, jt_out, judge_provider, judge_model)
    cost_usd = {
        "measured": True,
        "gen": round(gen_cost, 4),
        "judge": round(judge_cost, 4),
        "total": round(gen_cost + judge_cost, 4),
        "gen_tokens": {"input": gen_in, "output": gen_out, "provider": gen_provider, "model": gen_model},
        "judge_tokens": {"input": jt_in, "output": jt_out, "calls": jt_calls,
                         "provider": judge_provider, "model": judge_model},
        "note": "per-call token returns (cache-aware); priced via src/llm_cost. "
                "gen=evaporate backend ledger; judge=judge client usage delta over scoring.",
    }
    cost_columns = {
        "synthesis": (None if args.variant == "direct" else {
            "total_tokens": synth.get("total_tokens", 0),
            "calls": synth.get("calls", 0),
            "amortized_per_sampled_doc": round(synth.get("total_tokens", 0) / max(1, len(sample_docs_full)), 1),
        }),
        "apply": {
            "avg_cost_ratio_unsampled": round(mean(cost_ratios), 6) if cost_ratios else None,
            "extraction_llm_tokens": apply_extract.get("total_tokens", 0),
            "note": "apply cost_ratio = retrieved_token_count / _count_tokens(doc); "
                    "the only cross-variant-comparable cost number.",
        },
    }

    # ── Code+ weak-supervision aggregation rollup (acceptance #9) ──
    codeplus_ws = var_out.get("codeplus_ws", {}) or {}
    ws_rollup = None
    if args.variant == "codeplus" and codeplus_ws:
        reasons: dict[str, int] = {}
        for st in codeplus_ws.values():
            for r, n in (st.get("fallback_reasons", {}) or {}).items():
                reasons[r] = reasons.get(r, 0) + n
        ws_rollup = {
            "requested_combiner": args.combiner,
            "questions": len(codeplus_ws),
            "ws_applied_docs": sum(st.get("ws_applied", 0) for st in codeplus_ws.values()),
            "mv_fallback_docs": sum(st.get("mv_fallback", 0) for st in codeplus_ws.values()),
            "fallback_reasons": reasons,
            "per_question": codeplus_ws,
        }

    summary = {
        "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "baseline": "evaporate",
        "variant": args.variant,
        "combiner_mode": (args.combiner if args.variant == "codeplus" else None),
        "codeplus_ws": ws_rollup,
        "sampling_strategy": args.sampling_strategy,
        "dataset": args.dataset,
        "cluster": args.cluster,
        "judge_model": args.judge_model,
        "extraction_model": args.model,
        "num_questions": len(questions),
        "num_sampled_docs": len(sample_docs_full),
        "num_unsampled_docs": len(unsampled_docs_full),
        "selection_summary": {
            "keep_thresh": 0.5,
            "policy": ("upstream threshold when >=1 function clears keep_thresh; "
                       "else bypass threshold and force top-k functions by raw F1"),
            "n_questions": len(per_question_summary),
            "n_questions_threshold_bypass": sum(
                1 for q in per_question_summary if q.get("threshold_bypass_fallback")),
        },
        "questions": per_question_summary,
        "overall": {
            "avg_sampled_accuracy": round(mean(sampled_accs), 4) if sampled_accs else None,
            "avg_unsampled_accuracy": round(mean(unsampled_accs), 4) if unsampled_accs else None,
            "avg_combined_accuracy": round(mean(combined_accs), 4) if combined_accs else None,
            # unsampled-only, matches pipeline basis (do not rename: schema parity)
            "avg_cost_ratio": round(mean(cost_ratios), 6) if cost_ratios else None,
            # ONLINE (Table-6 analog): mean over questions of per-doc rule-application
            # latency (pure Python, no LLM). Matches Scout's Online (s/doc) metric.
            "avg_online_s_per_doc": (
                round(mean([q["online_s_per_doc"] for q in per_question_summary
                            if q.get("online_s_per_doc") is not None]), 6)
                if any(q.get("online_s_per_doc") is not None for q in per_question_summary) else None),
            "max_llm_calls_in_apply_window": max(
                [q.get("llm_calls_in_window") or 0 for q in per_question_summary], default=0),
        },
        "cost_columns": cost_columns,
        "cost_usd": cost_usd,
        "per_doc_field_notes": (
            "per-doc `latency_seconds` and `input_tokens` are always 0 for Evaporate "
            "(the gpt54mini ledger is global/phase-tagged, not cleanly attributable "
            "per doc) — do NOT compare these fields against pipeline runs; use "
            "`cost_columns` and the ledger for cost/latency."
        ),
        "ledger_summary": ledger_sum,
        "variant_errors": var_out.get("errors", []),
        "split_sources": {"sample": str(sample_labels_path), "unsampled": str(unsampled_labels_path)},
        "upstream_sha": _read_upstream_sha(),
        # ── C: where a reviewer finds the raw provenance (paths relative to output-dir) ──
        "artifacts": {
            "run_metadata": "run_metadata.json",
            "subprocess_log": f"logs/run_variant.{args.variant}.log",
            "variant_input": f"staging/{args.variant}/input.json",
            "variant_output": f"staging/{args.variant}/output.json",
            "variant_output_keys": {
                "functions": "synthesized function source per question_slug: functions[q_slug][*].source (+ score, from_doc)",
                "ledger": "full per-LLM-call token log (input/output/cached/reasoning)",
                "ledger_summary": "phase-tagged totals (synthesis vs extraction)",
                "codeplus_ws": "WS/MV aggregation stats per question_slug (ws_applied, mv_fallback, fallback_reasons)",
                "predictions": "raw per-doc predicted strings per question_slug",
            },
            "split": {
                "sample": str(sample_labels_path),
                "unsampled": str(unsampled_labels_path),
            },
            "per_question_eval": "{question_slug}_{sampled|unsampled}.json (per-doc predicted/ground_truth/correct/cost_ratio)",
            "per_question_raw": f"{{question_slug}}/{{sampled|unsampled}}/{rule_set_slug}.json",
        },
    }
    summary_path = apply_root / "pipeline_summary.json"
    pipeline._write_json(summary_path, summary)

    # ── A: rewrite metadata with completion status + timing (rc=0; failures keep
    # the 'running' record from launch plus the subprocess log) ──
    end_dt = datetime.now(timezone.utc)
    _write_run_metadata(apply_root, args, status="completed", end_dt=end_dt,
                        duration_s=time.time() - t0, subprocess_rc=0, **_meta_kw)

    print(f"\n=== Done. Summary -> {summary_path} ===", flush=True)
    print(f"    sampled_acc={summary['overall']['avg_sampled_accuracy']} "
          f"unsampled_acc={summary['overall']['avg_unsampled_accuracy']} "
          f"combined_acc={summary['overall']['avg_combined_accuracy']} "
          f"apply_cost_ratio={summary['overall']['avg_cost_ratio']}", flush=True)
    return summary


def _read_upstream_sha() -> str | None:
    return _git_info(Path(__file__).resolve().parent / "upstream").get("sha")


def _git_info(repo_dir: Path) -> dict:
    """Best-effort {sha, branch, dirty} for a repo — provenance for review.
    Any field is None if git is unavailable or the dir isn't a repo."""
    def _g(args: list[str]) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=str(repo_dir),
                                  capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None
    status = _g(["status", "--porcelain"])
    return {
        "sha": _g(["rev-parse", "HEAD"]) or None,
        "branch": _g(["rev-parse", "--abbrev-ref", "HEAD"]) or None,
        "dirty": (bool(status) if status is not None else None),
    }


def _write_run_metadata(apply_root: Path, args: argparse.Namespace, *, status: str,
                        start_dt: datetime, queries: list[str], queries_src: str,
                        processing_dir: str, sample_path: Any, unsampled_path: Any,
                        n_sampled: int | None = None, n_unsampled: int | None = None,
                        end_dt: datetime | None = None, duration_s: float | None = None,
                        subprocess_rc: int | None = None) -> dict:
    """A: launch + provenance record. Written once at launch (status='running')
    and rewritten at completion (status='completed') with timing + doc counts."""
    meta = {
        "baseline": "evaporate",
        "status": status,
        "variant": args.variant,
        "combiner": (args.combiner if args.variant == "codeplus" else None),
        "dataset": args.dataset,
        "cluster": args.cluster,
        "sampling_strategy": args.sampling_strategy,
        "extraction_model": args.model,
        "judge_model": args.judge_model,
        "argv": sys.argv,
        "resolved_args": vars(args),
        "git": {"main_repo": _git_info(_ROOT), "upstream": _git_info(_HERE_UPSTREAM)},
        "python": sys.executable,
        "venv_python": str(_VENV_PY),
        "platform": platform.platform(),
        "queries_file": queries_src,
        "num_queries": len(queries),
        "processing_dir": processing_dir,
        "split_sources": {"sample": str(sample_path), "unsampled": str(unsampled_path)},
        "num_sampled_docs": n_sampled,
        "num_unsampled_docs": n_unsampled,
        "smoke_limits": {
            "limit_questions": args.limit_questions or None,
            "limit_sampled": args.limit_sampled or None,
            "limit_unsampled": args.limit_unsampled or None,
        },
        "start_time": start_dt.isoformat(),
        "end_time": end_dt.isoformat() if end_dt else None,
        "duration_seconds": (round(duration_s, 1) if duration_s is not None else None),
        "subprocess_returncode": subprocess_rc,
    }
    pipeline._write_json(apply_root / "run_metadata.json", meta)
    return meta


def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaporate baseline (fair vs LSF llm-rule-gen)")
    ap.add_argument("--dataset", required=True, choices=["financebench", "court", "nopv", "officeqa", "product", "tropic"])
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--cluster", default="single_cluster")
    ap.add_argument("--sampling-strategy", default="random", choices=["random", "fps"])
    ap.add_argument("--queries", default=None, help="override queries file (default: data/<dataset>/queries.json)")
    ap.add_argument("--processing-dir", default=None)
    ap.add_argument("--model", default="gpt54", choices=["gpt54", "gpt54mini"],
                    help="Evaporate synthesis/extraction/GOLD_KEY model — MUST match the LSF "
                         "variant being compared (default gpt54 = LSF's default rule-gen model)")
    ap.add_argument("--judge-model", default="gpt54", help="judge model module (gpt54; NOT mini)")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the judge/accuracy phase (latency-only run; avoids judge "
                         "LLM calls competing with synthesis for the provider rate limit)")
    ap.add_argument("--judge-workers", type=int, default=8,
                    help="concurrent judge API calls per split (independent per-doc; 1 = sequential)")
    ap.add_argument("--question-workers", type=int, default=1,
                    help="run N questions concurrently as separate run_variant subprocesses "
                         "(synthesis is the serial bottleneck; questions are independent). 1 = sequential.")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--timeout", type=int, default=7200, help="run_variant subprocess timeout (s)")
    # variant knobs
    ap.add_argument("--chunk-chars", type=int, default=3000)  # upstream configs.py default
    ap.add_argument("--num-functions", type=int, default=10)
    ap.add_argument("--topk", type=int, default=10)  # upstream num_top_k_scripts default
    ap.add_argument("--max-extract-chunks", type=int, default=40, help="Direct: cap chunks/doc")
    ap.add_argument("--combiner", default="ws", choices=["ws", "mv"],
                    help="Code+ aggregation: ws (snorkel LabelModel, default) | mv (majority-vote fallback)")
    # smoke controls
    ap.add_argument("--limit-questions", type=int, default=0)
    ap.add_argument("--limit-sampled", type=int, default=0,
                    help="SMOKE ONLY: cap synthesis docs (shrinks the sampled set; not a fair split)")
    ap.add_argument("--limit-unsampled", type=int, default=0)
    return ap


if __name__ == "__main__":
    run(_build_cli().parse_args())
