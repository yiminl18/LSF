"""Evaluate a baseline strategy across all questions × all docs in a split.

Output layout:
    baseline_results/<dataset>/<baseline>_<model>/
        <question_slug>/<doc_name>.json    # one file per (question, doc) pair
        summary.json                        # mean accuracy, tokens, latency

Usage:
    # Agentic Claude QA on financebench (default dataset), sampled docs
    python src/baseline/run_eval.py --baseline agentic_claude_qa --model opus47 --split sampled

    # MDocAgent on nopv (PDF-only dataset)
    python src/baseline/run_eval.py --baseline agentic_mdocagent --model gpt54 --dataset nopv \
        --max-docs 3

    # Single question (by slug prefix)
    python src/baseline/run_eval.py --baseline agentic_claude_qa --model opus47 --split sampled \
        --question-slug what_is_the_registrants_telephone_number
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import random
import re
import sys
import warnings
from pathlib import Path
from statistics import mean
from typing import Callable

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import time

from baseline.cost import compute_cost as _compute_cost

# Logical model name used to look up judge pricing (the judge always runs on
# Azure gpt-5.4 regardless of the baseline's gen model).
_JUDGE_LOGICAL_MODEL = "gpt-5.4"
_JUDGE_PROVIDER = "azure"

_FINANCEBENCH_PROCESSING_DIR = _ROOT / "data/financebench/processing"
_FINANCEBENCH_QUERIES_FILE   = _ROOT / "data/financebench/sample_queries.txt"

_FINANCEBENCH_SPLIT_LABELS = {
    "sampled":   _ROOT / "data/financebench/sample/single_cluster/random/sample_doc_labels.json",
    "unsampled": _ROOT / "data/financebench/sample/single_cluster/random/unsampled_doc_labels.json",
}

_NOPV_ROOT             = _ROOT / "data/nopv"
_NOPV_QUERIES_FILE     = _NOPV_ROOT / "queries.json"
_NOPV_RAW_DIR          = _NOPV_ROOT / "raw"
_NOPV_ALL_LABELS_FILE  = _NOPV_ROOT / "all_labels.json"

_COURT_ROOT             = _ROOT / "data/court"
_COURT_QUERIES_FILE     = _COURT_ROOT / "queries.json"
_COURT_ALL_LABELS_FILE  = _COURT_ROOT / "all_labels.json"
# Court PDFs live under datasets/, not data/, so they can be browsed independently of labels.
_COURT_RAW_DIR          = _ROOT / "datasets" / "court" / "latest" / "raw"

_OFFICEQA_ROOT             = _ROOT / "data/officeqa"
_OFFICEQA_QUERIES_FILE     = _OFFICEQA_ROOT / "queries.json"
_OFFICEQA_ALL_LABELS_FILE  = _OFFICEQA_ROOT / "all_labels.json"
# officeqa source data ships parsed_json (per-page element schema) under the
# repo-external datasets/ tree. MDocAgent's adapter reads .json directly — no
# PDF synthesis needed since image_agent runs through NoOpModel.
_OFFICEQA_PARSED_JSON_DIR  = _ROOT / "datasets/officeqa/latest/parsed_json"


class DatasetSpec:
    """All dataset-specific resolutions in one place: questions, labels, paths."""

    def __init__(
        self,
        *,
        name: str,
        questions: list[str],
        labels: dict[str, dict],
        doc_path_for: Callable[[str], Path],
        labels_file_path: Path,
        doc_kind: str,
    ) -> None:
        self.name = name
        self.questions = questions
        self.labels = labels
        self.doc_path_for = doc_path_for
        self.labels_file_path = labels_file_path
        self.doc_kind = doc_kind  # "json" or "pdf"

def _load_financebench(args) -> DatasetSpec:
    questions = [
        l.strip() for l in _FINANCEBENCH_QUERIES_FILE.read_text().splitlines() if l.strip()
    ]
    labels_path = (
        Path(args.labels_file) if args.labels_file else _FINANCEBENCH_SPLIT_LABELS[args.split]
    )
    labels: dict[str, dict] = json.loads(labels_path.read_text(encoding="utf-8"))

    def doc_path_for(doc_name: str) -> Path:
        return _FINANCEBENCH_PROCESSING_DIR / f"{doc_name}_reconstructed.json"

    return DatasetSpec(
        name="financebench",
        questions=questions,
        labels=labels,
        doc_path_for=doc_path_for,
        labels_file_path=labels_path,
        doc_kind="json",
    )


def _load_nopv(args) -> DatasetSpec:
    """Load nopv labels (financebench shape) produced by data/nopv/generate_labels.py."""
    queries: list[dict] = json.loads(_NOPV_QUERIES_FILE.read_text(encoding="utf-8"))
    questions = [q["text"] for q in queries]

    if not _NOPV_ALL_LABELS_FILE.exists():
        raise FileNotFoundError(
            f"nopv labels not found at {_NOPV_ALL_LABELS_FILE}. "
            "Run `python data/nopv/generate_labels.py` to materialise GT for the "
            "current queries.json."
        )
    labels: dict[str, dict] = json.loads(_NOPV_ALL_LABELS_FILE.read_text(encoding="utf-8"))

    def doc_path_for(doc_name: str) -> Path:
        return _NOPV_RAW_DIR / f"{doc_name}.pdf"

    return DatasetSpec(
        name="nopv",
        questions=questions,
        labels=labels,
        doc_path_for=doc_path_for,
        labels_file_path=_NOPV_ALL_LABELS_FILE,
        doc_kind="pdf",
    )


def _load_court(args) -> DatasetSpec:
    """Load court labels (PDF-only baselines look up raw PDFs under datasets/court/)."""
    queries: list[dict] = json.loads(_COURT_QUERIES_FILE.read_text(encoding="utf-8"))
    questions = [q["text"] for q in queries]

    if not _COURT_ALL_LABELS_FILE.exists():
        raise FileNotFoundError(
            f"court labels not found at {_COURT_ALL_LABELS_FILE}. "
            "Pull data/court/ from origin/yiming-dev or run data/court/generate_labels.py."
        )
    labels: dict[str, dict] = json.loads(_COURT_ALL_LABELS_FILE.read_text(encoding="utf-8"))

    def doc_path_for(doc_name: str) -> Path:
        return _COURT_RAW_DIR / f"{doc_name}.pdf"

    return DatasetSpec(
        name="court",
        questions=questions,
        labels=labels,
        doc_path_for=doc_path_for,
        labels_file_path=_COURT_ALL_LABELS_FILE,
        doc_kind="pdf",
    )


def _load_officeqa(args) -> DatasetSpec:
    """Load officeqa labels; MDocAgent ingests parsed_json directly.

    Honours ``--labels-file`` (e.g. plan-D subsets at
    ``data/officeqa/all_labels_planD.json``). Questions are intersected with
    whatever labels survive so dropped query texts don't show up in the run.
    """
    queries: list[dict] = json.loads(_OFFICEQA_QUERIES_FILE.read_text(encoding="utf-8"))

    labels_path = Path(args.labels_file) if args.labels_file else _OFFICEQA_ALL_LABELS_FILE
    if not labels_path.exists():
        raise FileNotFoundError(
            f"officeqa labels not found at {labels_path}. "
            "Pull data/officeqa/ from origin/yiming-dev, or generate a subset "
            "via data/officeqa/make_plan_d.py."
        )
    labels: dict[str, dict] = json.loads(labels_path.read_text(encoding="utf-8"))

    # Intersect queries with the question keys actually present in the labels
    # so subset files (e.g. plan-D) don't ask questions whose ground truth was
    # dropped by the filter.
    present_questions: set[str] = set()
    for doc_labels in labels.values():
        present_questions.update(doc_labels.keys())
    questions = [q["text"] for q in queries if q["text"] in present_questions]

    def doc_path_for(doc_name: str) -> Path:
        return _OFFICEQA_PARSED_JSON_DIR / f"{doc_name}.json"

    return DatasetSpec(
        name="officeqa",
        questions=questions,
        labels=labels,
        doc_path_for=doc_path_for,
        labels_file_path=labels_path,
        # "pdf" gates non-PDF-aware baselines from this dataset via the
        # SUPPORTS_PDF_INPUT check; mdocagent declares it and accepts both
        # .pdf and .json inputs via the adapter dispatch.
        doc_kind="pdf",
    )


_DATASET_LOADERS: dict[str, Callable[[argparse.Namespace], DatasetSpec]] = {
    "financebench": _load_financebench,
    "nopv": _load_nopv,
    "court": _load_court,
    "officeqa": _load_officeqa,
}


def _run_baseline(baseline_mod, kwargs: dict) -> dict:
    """Call run_qa while only passing kwargs the baseline can accept."""
    sig = inspect.signature(baseline_mod.run_qa)
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if accepts_kwargs:
        return baseline_mod.run_qa(**kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return baseline_mod.run_qa(**accepted)


def _resolve_dataset(args) -> DatasetSpec:
    loader = _DATASET_LOADERS.get(args.dataset)
    if loader is None:
        raise ValueError(f"Unknown --dataset: {args.dataset!r}")
    return loader(args)


_JUDGE_SYSTEM = """\
You are an answer equivalence judge for a financial document QA system.
Judge whether the predicted answer is semantically equivalent to the ground truth.
Equivalence rules:
- Treat "2017" and "year 2017" as the same
- Treat "$4.5 billion" and "4,500 million" as the same if numerically equal
- Ignore leading/trailing whitespace, punctuation, and capitalization differences
- If the predicted answer is "NOT_FOUND", "NOT FOUND", or null, always judge as incorrect
Reply with exactly one word: CORRECT or INCORRECT"""


def _make_slug(q: str) -> str:
    s = q.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60]


def _active_question_slugs(questions: list[str], question_slug_prefix: str | None) -> list[str]:
    slugs: list[str] = []
    for question in questions:
        slug = _make_slug(question)
        if question_slug_prefix and not slug.startswith(question_slug_prefix):
            continue
        slugs.append(slug)
    return slugs


def _completed_docs_for_questions(out_base: Path, question_slugs: list[str]) -> set[str]:
    if not question_slugs:
        return set()

    per_question_docs: list[set[str]] = []
    for slug in question_slugs:
        q_dir = out_base / slug
        docs = {p.stem for p in q_dir.glob("*.json")} if q_dir.exists() else set()
        per_question_docs.append(docs)

    return set.intersection(*per_question_docs) if per_question_docs else set()


def _judge(question: str, ground_truth, predicted, gpt54_mod) -> dict:
    """Return a dict with verdict + token/cost/latency telemetry for the judge call.

    Short-circuits without an API call when either side is missing.
    """
    if ground_truth is None or predicted is None:
        return {
            "verdict": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_seconds": 0.0,
            "cost_usd": 0.0,
            "model": _JUDGE_LOGICAL_MODEL,
            "raw_verdict": None,
        }
    gt_str   = json.dumps(ground_truth) if not isinstance(ground_truth, str) else ground_truth
    pred_str = str(predicted)

    t0 = time.time()
    resp = gpt54_mod.client.chat.completions.create(
        model=gpt54_mod.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user",   "content": f"Question: {question}\nGround Truth: {gt_str}\nPredicted: {pred_str}"},
        ],
        max_completion_tokens=10,
        temperature=0.0,
    )
    latency = round(time.time() - t0, 3)

    raw_verdict = (resp.choices[0].message.content or "").strip()
    verdict_lc = raw_verdict.lower()
    if verdict_lc not in ("correct", "incorrect"):
        warnings.warn(f"Unexpected judge verdict: {raw_verdict!r}")

    usage = resp.usage
    in_tok  = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    out_tok = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    try:
        cost_usd = _compute_cost(in_tok, out_tok, _JUDGE_PROVIDER, _JUDGE_LOGICAL_MODEL)
    except Exception:
        cost_usd = 0.0

    return {
        "verdict": verdict_lc == "correct",
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "latency_seconds": latency,
        "cost_usd": cost_usd,
        "model": _JUDGE_LOGICAL_MODEL,
        "raw_verdict": raw_verdict,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run baseline eval across all questions × docs")
    ap.add_argument("--baseline", default="agentic_claude_qa",
                    help="Baseline module name under src/baseline/ (default: agentic_claude_qa)")
    ap.add_argument("--model",    default="opus47",
                    help="Model alias (default: opus47)")
    ap.add_argument("--dataset",  default="financebench",
                    choices=tuple(_DATASET_LOADERS.keys()),
                    help="Dataset to evaluate (default: financebench)")
    ap.add_argument("--split",    choices=("sampled", "unsampled"), default="sampled",
                    help="financebench split selector; ignored for PDF-only datasets")
    ap.add_argument("--labels-file", default=None,
                    help="Optional path to a labels JSON file; overrides --split")
    ap.add_argument("--question-slug", default=None,
                    help="Run only questions whose slug starts with this prefix")
    ap.add_argument("--output-name", default=None,
                    help="Optional custom output directory name under baseline_results/<dataset>/")
    ap.add_argument("--max-docs", type=int, default=None,
                    help="Run at most N not-yet-completed docs from the selected labels set")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for sampling docs when --max-docs is set (default: 0)")
    ap.add_argument("--skip-docs-in", default=None, action="append",
                    help="Path to an existing results dir; skip any doc already run there (repeatable)")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--timeout",  type=int, default=300)
    ap.add_argument("--max-pages", type=int, default=None,
                    help="Optional page cap for PDF baselines that support it")
    ap.add_argument("--provider", default="azure",
                    help="Reader provider for baselines that use direct LLM calls")
    args = ap.parse_args()

    baseline_mod = importlib.import_module(f"baseline.{args.baseline}")
    gpt54_mod    = importlib.import_module("models.gpt54")

    spec = _resolve_dataset(args)

    if spec.doc_kind == "pdf" and not getattr(baseline_mod, "SUPPORTS_PDF_INPUT", False):
        ap.error(
            f"baseline {args.baseline!r} does not declare SUPPORTS_PDF_INPUT=True "
            f"but dataset {spec.name!r} only ships PDFs."
        )

    questions   = spec.questions
    labels      = spec.labels
    labels_path = spec.labels_file_path
    split_label = args.split if spec.name == "financebench" else "all"

    out_name = args.output_name or f"{args.baseline}_{args.model}"
    out_base = _ROOT / "baseline_results" / spec.name / out_name
    out_base.mkdir(parents=True, exist_ok=True)

    active_slugs = _active_question_slugs(questions, args.question_slug)
    completed_docs = _completed_docs_for_questions(out_base, active_slugs)

    # Collect additional docs to skip from one or more existing results directories
    skip_docs: set[str] = set()
    for skip_path in (args.skip_docs_in or []):
        skip_dir = Path(skip_path)
        found = {p.stem for p in skip_dir.rglob("*.json") if p.name != "summary.json"}
        skip_docs.update(found)
        print(f"skip_docs_in={skip_dir}  found={len(found)}  total_skip={len(skip_docs)}")

    selected_items = sorted(labels.items())
    if args.max_docs is not None:
        # Filter out completed docs first, then randomly sample max_docs from
        # the remaining pool with the user-provided seed. We re-sort the
        # sample alphabetically so iteration order is stable across reruns
        # with the same seed — the seed only determines *which* docs get
        # picked, not the order they're processed in.
        eligible = [
            (pdf_key, doc_labels)
            for pdf_key, doc_labels in selected_items
            if pdf_key.replace(".pdf", "") not in completed_docs
        ]
        rng = random.Random(args.seed)
        k = min(args.max_docs, len(eligible))
        selected_items = sorted(rng.sample(eligible, k))
        labels = dict(selected_items)

    if args.max_docs is not None:
        metadata = {
            "baseline": args.baseline,
            "model": args.model,
            "dataset": spec.name,
            "split": split_label,
            "labels_file": str(labels_path.relative_to(_ROOT)) if labels_path.is_relative_to(_ROOT) else str(labels_path),
            "output_name": out_name,
            "question_slug_prefix": args.question_slug,
            "max_docs": args.max_docs,
            "max_pages": args.max_pages,
            "seed": args.seed,
            "provider": args.provider,
            "selected_docs": [pdf_key.replace(".pdf", "") for pdf_key, _ in selected_items],
            "completed_docs_skipped": sorted(completed_docs),
        }
        (out_base / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(f"baseline={args.baseline}  model={args.model}  dataset={spec.name}  split={split_label}")
    print(f"questions={len(questions)}  docs={len(labels)}  output={out_base}")
    print(f"labels_file={labels_path}")
    if args.max_docs is not None:
        print(f"completed_docs_skipped={len(completed_docs)}  selected_docs={len(labels)}")
    print()

    all_summaries: list[dict] = []

    for question in questions:
        slug = _make_slug(question)
        if args.question_slug and not slug.startswith(args.question_slug):
            continue

        q_dir = out_base / slug
        q_dir.mkdir(exist_ok=True)

        per_doc_results: list[dict] = []

        for pdf_key, doc_labels in labels.items():
            doc_name  = pdf_key.replace(".pdf", "")
            doc_path  = spec.doc_path_for(doc_name)
            out_file  = q_dir / f"{doc_name}.json"

            if doc_name in skip_docs:
                continue

            if args.skip_existing and out_file.exists():
                try:
                    per_doc_results.append(json.loads(out_file.read_text(encoding="utf-8")))
                except Exception:
                    pass
                continue

            if not doc_path.exists():
                print(f"  SKIP (no {spec.doc_kind.upper()}): {doc_name}")
                continue

            ground_truth = doc_labels.get(question)

            try:
                baseline_kwargs = {
                    "doc_path": doc_path,
                    "question": question,
                    "model": args.model,
                    "timeout": args.timeout,
                    "log_dir": q_dir / "logs",
                    "log_stem": doc_name,
                    "provider": args.provider,
                }
                if args.max_pages is not None:
                    baseline_kwargs["max_pages"] = args.max_pages
                result = _run_baseline(baseline_mod, baseline_kwargs)
            except Exception as e:
                print(f"  ERROR {doc_name}: {e}")
                result = {
                    "status": "error",
                    "answer": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_seconds": 0.0,
                    "total_cost_usd": None,
                    "model": args.model,
                }

            judge = _judge(question, ground_truth, result.get("answer"), gpt54_mod)

            gen_in   = int(result.get("input_tokens", 0) or 0)
            gen_out  = int(result.get("output_tokens", 0) or 0)
            gen_lat  = float(result.get("latency_seconds", 0.0) or 0.0)
            gen_cost = result.get("total_cost_usd")
            gen_cost_num = float(gen_cost or 0.0)

            judge_cost_num = float(judge["cost_usd"] or 0.0)
            total_cost = (gen_cost_num + judge_cost_num) if gen_cost is not None else None
            total_latency = round(gen_lat + judge["latency_seconds"], 3)

            record = {
                "doc_name":              doc_name,
                "question":              question,
                "question_slug":         slug,
                "split":                 split_label,
                "ground_truth":          ground_truth,
                "answer":                result.get("answer"),
                "correct":               judge["verdict"],
                "status":                result.get("status"),
                "model":                 result.get("model", args.model),
                # gen-side telemetry
                "gen_input_tokens":      gen_in,
                "gen_output_tokens":     gen_out,
                "gen_latency_seconds":   gen_lat,
                "gen_cost_usd":          gen_cost,
                # judge-side telemetry
                "judge_model":           judge["model"],
                "judge_input_tokens":    judge["input_tokens"],
                "judge_output_tokens":   judge["output_tokens"],
                "judge_latency_seconds": judge["latency_seconds"],
                "judge_cost_usd":        judge["cost_usd"],
                "judge_verdict_raw":     judge["raw_verdict"],
                # totals
                "total_cost_usd":        total_cost,
                "total_latency_seconds": total_latency,
                # legacy keys (kept for downstream compatibility)
                "input_tokens":          gen_in,
                "output_tokens":         gen_out,
                "latency_seconds":       gen_lat,
            }
            for k, v in result.items():
                if k not in record and k != "answer":
                    record[k] = v
            out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False),
                                encoding="utf-8")
            per_doc_results.append(record)
            mark = "✓" if judge["verdict"] else "✗"
            print(
                f"  {mark} {doc_name}: {record['answer']!r}  "
                f"gen={gen_lat:.1f}s/${gen_cost_num:.4f}  "
                f"judge={judge['latency_seconds']:.1f}s/${judge_cost_num:.4f}"
            )

        if not per_doc_results:
            continue

        n         = len(per_doc_results)
        n_correct = sum(r["correct"] for r in per_doc_results)
        accuracy  = round(n_correct / n, 4)

        def _mean_field(field: str) -> float:
            vals: list[float] = []
            for r in per_doc_results:
                v = r.get(field)
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    continue
            return round(mean(vals), 4) if vals else 0.0

        avg_gen_in   = _mean_field("gen_input_tokens")
        avg_gen_out  = _mean_field("gen_output_tokens")
        avg_gen_lat  = _mean_field("gen_latency_seconds")
        avg_gen_cost = _mean_field("gen_cost_usd")
        avg_jud_in   = _mean_field("judge_input_tokens")
        avg_jud_out  = _mean_field("judge_output_tokens")
        avg_jud_lat  = _mean_field("judge_latency_seconds")
        avg_jud_cost = _mean_field("judge_cost_usd")
        avg_tot_cost = _mean_field("total_cost_usd")
        avg_tot_lat  = _mean_field("total_latency_seconds")

        q_summary = {
            "question":                  question,
            "question_slug":             slug,
            "split":                     split_label,
            "model":                     args.model,
            "judge_model":               _JUDGE_LOGICAL_MODEL,
            "n":                         n,
            "n_correct":                 n_correct,
            "accuracy":                  accuracy,
            "avg_gen_input_tokens":      avg_gen_in,
            "avg_gen_output_tokens":     avg_gen_out,
            "avg_gen_latency_seconds":   avg_gen_lat,
            "avg_gen_cost_usd":          avg_gen_cost,
            "avg_judge_input_tokens":    avg_jud_in,
            "avg_judge_output_tokens":   avg_jud_out,
            "avg_judge_latency_seconds": avg_jud_lat,
            "avg_judge_cost_usd":        avg_jud_cost,
            "avg_total_cost_usd":        avg_tot_cost,
            "avg_total_latency_seconds": avg_tot_lat,
            # back-compat keys for downstream readers
            "avg_input_tokens":          avg_gen_in,
            "avg_output_tokens":         avg_gen_out,
            "avg_latency_seconds":       avg_gen_lat,
        }
        all_summaries.append(q_summary)
        print(f"\nQuestion: {question[:70]}")
        print(
            f"  accuracy={accuracy:.2f} ({n_correct}/{n})  "
            f"gen_lat={avg_gen_lat:.1f}s  judge_lat={avg_jud_lat:.1f}s  "
            f"gen_cost=${avg_gen_cost:.4f}  judge_cost=${avg_jud_cost:.4f}  "
            f"total=${avg_tot_cost:.4f}\n"
        )

    summary_path = out_base / "summary.json"
    # Merge with any existing summary entries for other splits/runs
    existing: list[dict] = []
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    merged = {e["question_slug"]: e for e in existing}
    for s in all_summaries:
        merged[s["question_slug"]] = s
    summary_path.write_text(
        json.dumps(list(merged.values()), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if all_summaries:
        print(f"\n{'='*60}")
        print(f"  mean accuracy   : {mean(s['accuracy'] for s in all_summaries):.4f}")
        print(f"  mean gen tokens : in={mean(s['avg_gen_input_tokens']  for s in all_summaries):.0f}  "
              f"out={mean(s['avg_gen_output_tokens'] for s in all_summaries):.0f}")
        print(f"  mean gen lat    : {mean(s['avg_gen_latency_seconds']  for s in all_summaries):.1f}s")
        print(f"  mean judge lat  : {mean(s['avg_judge_latency_seconds'] for s in all_summaries):.2f}s")
        print(f"  mean gen cost   : ${mean(s['avg_gen_cost_usd']   for s in all_summaries):.4f}")
        print(f"  mean judge cost : ${mean(s['avg_judge_cost_usd'] for s in all_summaries):.4f}")
        print(f"  mean total cost : ${mean(s['avg_total_cost_usd'] for s in all_summaries):.4f}")
        print(f"  results in      : {out_base}/")


if __name__ == "__main__":
    main()
