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
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pipeline  # noqa: E402  (the comparison target; we reuse its seams)

_RUNTIME = Path(__file__).resolve().parent / "runtime"
_VENV_PY = _RUNTIME / ".venv-evaporate" / "bin" / "python"
_RUN_VARIANT = _RUNTIME / "run_variant.py"

VARIANTS = ("direct", "code", "codeplus")


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


def _resolve_queries(dataset: str, queries_file: str | None) -> list[str]:
    if queries_file:
        return pipeline._load_queries(queries_file)
    rel = _DEFAULT_QUERIES.get(dataset)
    cand = (_ROOT / rel) if rel else (_ROOT / "data" / dataset / "queries.json")
    if cand.exists():
        return pipeline._load_queries(str(cand))
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


def _run_variant_subprocess(input_path: Path, output_path: Path, timeout: int) -> dict:
    if not _VENV_PY.exists():
        raise FileNotFoundError(
            f"isolated venv python not found at {_VENV_PY}. "
            f"Run: bash {_RUNTIME / 'setup_env.sh'}"
        )
    cmd = [str(_VENV_PY), str(_RUN_VARIANT), "--input", str(input_path), "--output", str(output_path)]
    print(f"  [evaporate] subprocess: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"run_variant failed (rc={proc.returncode}).\n"
            f"STDOUT tail:\n{proc.stdout[-1500:]}\nSTDERR tail:\n{proc.stderr[-1500:]}"
        )
    if not output_path.exists():
        raise RuntimeError(f"run_variant returned 0 but no output at {output_path}")
    return json.loads(output_path.read_text(encoding="utf-8"))


# ── scoring (reuses pipeline's judge + cost denominator) ─────────────────────

def _score_split(
    *, variant: str, split: str, question: str, question_slug: str,
    doc_map: dict, labels: dict, var_preds: dict, judge_mod, apply_root: Path,
    rule_set_slug: str,
) -> dict:
    """Build pipeline-schema per-doc records + aggregate for one (question, split)."""
    run_dir = apply_root / question_slug / split
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_records: list[dict] = []
    per_doc: list[dict] = []

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

        correct = pipeline._judge(judge_mod, question, gt, predicted)

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

    processing_dir = args.processing_dir or str(pipeline._default_processing_dir(args.dataset))
    queries = _resolve_queries(args.dataset, args.queries)
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
    input_path = _stage_input(
        variant=args.variant, questions=questions, sample_docs=sample_text,
        all_docs=all_text, sampled_gold=sampled_gold, cfg=cfg, staging_dir=staging_dir,
    )
    output_path = staging_dir / "output.json"
    var_out = _run_variant_subprocess(input_path, output_path, timeout=args.timeout)

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
    judge_mod = __import__("importlib").import_module(f"models.{args.judge_model}")
    assert pipeline._JUDGE_SYSTEM, "pipeline judge system prompt missing"

    per_question_summary: list[dict] = []
    for q in questions:
        q_slug, question = q["slug"], q["text"]
        preds = var_out.get("predictions", {}).get(q_slug, {})
        sampled_eval = _score_split(
            variant=args.variant, split="sampled", question=question, question_slug=q_slug,
            doc_map=sample_docs_full, labels=sample_labels,
            var_preds={dn: preds.get(dn, {}) for dn in sample_docs_full},
            judge_mod=judge_mod, apply_root=apply_root, rule_set_slug=rule_set_slug,
        )
        unsampled_eval = _score_split(
            variant=args.variant, split="unsampled", question=question, question_slug=q_slug,
            doc_map=unsampled_docs_full, labels=unsampled_labels,
            var_preds={dn: preds.get(dn, {}) for dn in unsampled_docs_full},
            judge_mod=judge_mod, apply_root=apply_root, rule_set_slug=rule_set_slug,
        )
        per_question_summary.append({
            "question": question,
            "question_slug": q_slug,
            "num_functions": len(var_out.get("functions", {}).get(q_slug, [])) or None,
            "sampled_accuracy": sampled_eval.get("accuracy"),
            "unsampled_accuracy": unsampled_eval.get("accuracy"),
            "avg_cost_ratio_sampled": sampled_eval.get("avg_cost_ratio"),
            "avg_cost_ratio_unsampled": unsampled_eval.get("avg_cost_ratio"),
        })

    sampled_accs = [q["sampled_accuracy"] for q in per_question_summary if q.get("sampled_accuracy") is not None]
    unsampled_accs = [q["unsampled_accuracy"] for q in per_question_summary if q.get("unsampled_accuracy") is not None]
    # Unsampled-only basis, matching pipeline.py's `overall.avg_cost_ratio`
    # (pipeline.py aggregates `avg_cost_ratio_unsampled`); same field name on purpose.
    cost_ratios = [q["avg_cost_ratio_unsampled"] for q in per_question_summary if q.get("avg_cost_ratio_unsampled") is not None]

    # ── Gap C: two-column cost from the phase-tagged ledger ──
    ledger_sum = var_out.get("ledger_summary", {})
    by_phase = ledger_sum.get("by_phase", {})
    synth = by_phase.get("synthesis", {})
    apply_extract = by_phase.get("extraction", {})
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
        "questions": per_question_summary,
        "overall": {
            "avg_sampled_accuracy": round(mean(sampled_accs), 4) if sampled_accs else None,
            "avg_unsampled_accuracy": round(mean(unsampled_accs), 4) if unsampled_accs else None,
            # unsampled-only, matches pipeline basis (do not rename: schema parity)
            "avg_cost_ratio": round(mean(cost_ratios), 6) if cost_ratios else None,
        },
        "cost_columns": cost_columns,
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
    }
    summary_path = apply_root / "pipeline_summary.json"
    pipeline._write_json(summary_path, summary)
    print(f"\n=== Done. Summary -> {summary_path} ===", flush=True)
    print(f"    unsampled_acc={summary['overall']['avg_unsampled_accuracy']} "
          f"sampled_acc={summary['overall']['avg_sampled_accuracy']} "
          f"apply_cost_ratio={summary['overall']['avg_cost_ratio']}", flush=True)
    return summary


def _read_upstream_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent / "upstream"),
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaporate baseline (fair vs LSF llm-rule-gen)")
    ap.add_argument("--dataset", required=True, choices=["financebench", "court", "nopv"])
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
