"""End-to-end LSF pipeline: sampling -> rule gen -> (optional) refine -> apply + eval.

Each of the four stages exposes a strategy choice that maps to a documented
approach in docs/approach/*.md. See docs/pipeline.md for the full design.

Usage:

    from pipeline import run_pipeline

    result = run_pipeline(
        sampling_strategy = "random",
        rule_gen_strategy = "llm_coarse",
        refine_strategy   = "p_mini",
        apply_strategy    = "default",
        dataset           = "financebench",
        cluster           = "single_cluster",
        output_dir        = "results/e2e",
    )

Or via CLI:

    python src/pipeline.py --rule-gen-strategy llm_coarse --refine-strategy p_mini --apply-strategy default
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ── Strategy registries ─────────────────────────────────────────────────────

SAMPLING_STRATEGIES = ("random", "fps")
RULE_GEN_STRATEGIES = ("llm_coarse", "agent_langchain", "agent_claude", "agent_codex")
REFINE_STRATEGIES   = ("none", "v1", "p_mini", "p_gpt54", "p_proxy", "p_v2", "p_v3", "agentic", "agentic_codex")
APPLY_STRATEGIES    = ("merge", "default")


# ── Small helpers ───────────────────────────────────────────────────────────

def _make_slug(q: str) -> str:
    s = q.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60]


def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


def _load_docs(labels: dict, processing_dir: str) -> dict[str, dict]:
    docs: dict[str, dict] = {}
    for pdf_key in labels:
        doc_name = pdf_key.replace(".pdf", "")
        path = Path(processing_dir) / f"{doc_name}_reconstructed.json"
        if path.exists():
            docs[doc_name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            print(f"  WARN: missing {path}", flush=True)
    return docs


def _list_rule_names(folder: Path) -> list[str]:
    if not folder.is_dir():
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(folder)
        if f.startswith("rule_") and f.endswith(".py")
    )


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Stage 1 — Sampling ──────────────────────────────────────────────────────

def stage_sampling(
    *,
    strategy: str,
    dataset: str,
    cluster: str,
    output_dir: Path,
    skip_existing: bool,
) -> tuple[Path, Path]:
    """Return (sample_labels_path, unsampled_labels_path)."""
    base = _ROOT / "data" / dataset / "sample" / cluster

    if strategy == "random":
        sample    = base / "random" / "sample_doc_labels.json"
        unsampled = base / "random" / "unsampled_doc_labels.json"
        if not sample.exists() or not unsampled.exists():
            raise FileNotFoundError(
                f"random sample label files not found under {base / 'random'}. "
                f"Either pre-build them or switch --sampling-strategy fps."
            )
        return sample, unsampled

    if strategy == "fps":
        fps_dir = base / "fps"
        sample    = fps_dir / "sample_doc_labels.json"
        unsampled = fps_dir / "unsampled_doc_labels.json"
        if skip_existing and sample.exists() and unsampled.exists():
            print(f"  [sampling:fps] SKIP (exists at {fps_dir})", flush=True)
            return sample, unsampled
        # Reuse the FPS driver script; it's parameterised at module level for FinanceBench.
        # If you need fps for a different dataset, edit src/sampling/run_fps_sampling.py first.
        print(f"  [sampling:fps] running src/sampling/run_fps_sampling.py ...", flush=True)
        proc = subprocess.run(
            [sys.executable, "src/sampling/run_fps_sampling.py"],
            cwd=str(_ROOT), capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"FPS sampling failed:\n{proc.stderr[-2000:]}")
        if not sample.exists() or not unsampled.exists():
            raise FileNotFoundError(f"FPS finished but label files not at {fps_dir}")
        return sample, unsampled

    raise ValueError(f"Unknown sampling_strategy: {strategy!r}")


# ── Stage 2 — Rule Generation ──────────────────────────────────────────────

# Subprocess-based generators (agent_claude / agent_codex) operate per-question
# and don't expose the same callable interface as llm_coarse / agent_langchain.
_SUBPROCESS_GEN = {"agent_claude", "agent_codex"}


def stage_rule_gen(
    *,
    strategy: str,
    question: str,
    question_slug: str,
    sample_docs: list[dict],
    sample_doc_names: list[str],
    ground_truth: dict[str, Any],
    rules_dir: Path,
    output_dir: Path,
    model: str,
    skip_existing: bool,
) -> tuple[Path, dict]:
    """Generate rules for one question. Return (rule_folder, gen_metadata)."""

    rule_folder = rules_dir / f"{question_slug}_{len(sample_docs)}_{strategy}"
    rule_gen_out = output_dir / "rule_gen" / f"{question_slug}_rule_gen.json"

    if skip_existing and rule_folder.is_dir() and _list_rule_names(rule_folder):
        print(f"  [gen:{strategy}] SKIP {question_slug}", flush=True)
        return rule_folder, _read_json(rule_gen_out, default={})

    rule_folder.mkdir(parents=True, exist_ok=True)

    if strategy in {"llm_coarse", "agent_langchain"}:
        mod = importlib.import_module(f"rule_gen.{strategy}")
        fn  = next(v for k, v in vars(mod).items() if k.startswith("rule_gen_") and callable(v))
        result = fn(
            documents     = sample_docs,
            question      = question,
            ground_truth  = ground_truth,
            rules_dir     = str(rules_dir),
            output_dir    = str(output_dir / "rule_gen"),
            model_name    = model,
        )
        _write_json(rule_gen_out, result)
        print(f"  [gen:{strategy}] {question_slug}: {len(result.get('rules', []))} rules", flush=True)
        return rule_folder, result

    if strategy in _SUBPROCESS_GEN:
        mod_name = "agent_claude" if strategy == "agent_claude" else "agent_codex"
        cmd = [
            sys.executable, f"src/rule_gen/{mod_name}.py",
            question,
            "--docs", *sample_doc_names,
            "--rules-dir", str(rules_dir),
            "--model", model,
        ]
        proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"{strategy} failed for {question_slug}:\n{proc.stderr[-2000:]}")
        result = {"strategy": strategy, "stdout_tail": proc.stdout[-500:]}
        _write_json(rule_gen_out, result)
        print(f"  [gen:{strategy}] {question_slug}: done", flush=True)
        return rule_folder, result

    raise ValueError(f"Unknown rule_gen_strategy: {strategy!r}")


# ── Stage 3 — Rule Refinement ──────────────────────────────────────────────

def stage_refine(
    *,
    strategy: str,
    question: str,
    question_slug: str,
    rule_folder: Path,
    sample_docs: list[dict],
    ground_truth: dict[str, Any],
    rules_dir: Path,
    output_dir: Path,
    model: str,
    skip_existing: bool,
) -> Path:
    """Run refinement and return the *refined* rule folder. Caller passes this folder to Stage 4."""

    refined_dir = output_dir / "refined_rules"
    refined_folder = refined_dir / question_slug
    refine_out = output_dir / "rule_refine" / f"{question_slug}_refine.json"

    if skip_existing and refined_folder.is_dir() and _list_rule_names(refined_folder):
        print(f"  [refine:{strategy}] SKIP {question_slug}", flush=True)
        return refined_folder

    refined_folder.mkdir(parents=True, exist_ok=True)
    rule_names = _list_rule_names(rule_folder)

    if strategy == "v1":
        from rule_refine.v1 import rule_refine as _v1, evaluate_merge_accuracy
        model_mod = importlib.import_module(f"models.{model}")
        target_acc, *_ = evaluate_merge_accuracy(
            rule_names=rule_names, documents=sample_docs,
            ground_truth=ground_truth, question=question,
            rule_folder=rule_folder, model_mod=model_mod,
        )
        result = _v1(
            rule_names=rule_names, target_accuracy=target_acc,
            question=question, question_slug=question_slug,
            documents=sample_docs, ground_truth=ground_truth,
            rules_dir=str(rules_dir), output_dir=str(refined_dir),
        )
        _write_json(refine_out, result)
        return refined_folder

    if strategy in {"p_mini", "p_gpt54"}:
        # Same code, different MODEL_NAME global. We monkey-patch on import.
        mod = importlib.import_module("rule_refine.selection.select_rules_pareto")
        mod.MODEL_NAME = "gpt54mini" if strategy == "p_mini" else "gpt54"
        result = mod.select_rules_pareto(
            rule_names=rule_names, question=question, question_slug=question_slug,
            documents=sample_docs, ground_truth=ground_truth,
            rules_dir=str(rules_dir), output_dir=str(refined_dir),
        )
        _write_json(refine_out, result)
        return refined_folder

    if strategy == "p_proxy":
        from rule_refine.selection.select_rules_pareto_proxy import select_rules_pareto_proxy
        result = select_rules_pareto_proxy(
            rule_names=rule_names, question=question, question_slug=question_slug,
            documents=sample_docs, ground_truth=ground_truth,
            rules_dir=str(rules_dir), output_dir=str(refined_dir),
        )
        _write_json(refine_out, result)
        return refined_folder

    if strategy == "p_v2":
        from rule_refine.selection.select_rules_pareto_v2 import select_rules_pareto_v2
        result = select_rules_pareto_v2(
            rule_names=rule_names, question=question, question_slug=question_slug,
            documents=sample_docs, ground_truth=ground_truth,
            rules_dir=str(rules_dir), output_dir=str(refined_dir),
        )
        _write_json(refine_out, result)
        return refined_folder

    if strategy == "p_v3":
        from rule_refine.selection.select_rules_pareto_v3 import select_rules_pareto_v3
        result = select_rules_pareto_v3(
            rule_names=rule_names, question=question, question_slug=question_slug,
            documents=sample_docs, ground_truth=ground_truth,
            rules_dir=str(rules_dir), output_dir=str(refined_dir),
        )
        _write_json(refine_out, result)
        return refined_folder

    if strategy in {"agentic", "agentic_codex"}:
        # Driver script handles per-question subprocess + trace capture.
        driver = "src/rule_refine/agentic.py" if strategy == "agentic" else "src/rule_refine/agentic_codex.py"
        cmd = [
            sys.executable, driver,
            "--slug", question_slug,
            "--rules-dir", str(rules_dir),
            "--out-dir", str(refined_dir),
        ]
        proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"{strategy} refine failed for {question_slug}:\n{proc.stderr[-2000:]}")
        _write_json(refine_out, {"strategy": strategy, "stdout_tail": proc.stdout[-500:]})
        return refined_folder

    raise ValueError(f"Unknown refine_strategy: {strategy!r}")


# ── Stage 4 — Rule Application + Evaluation ────────────────────────────────

_JUDGE_SYSTEM = (
    "You are an answer equivalence judge. You will be given a question, a "
    "predicted answer, and a ground truth answer. Judge whether the predicted "
    "answer is semantically equivalent to the ground truth.\n\n"
    "Reply with exactly one word: CORRECT or INCORRECT"
)


def _judge(model_mod, question: str, ground_truth, predicted) -> bool:
    if ground_truth is None or predicted is None:
        return False
    gt_str   = json.dumps(ground_truth) if not isinstance(ground_truth, str) else ground_truth
    pred_str = str(predicted)
    resp = model_mod.client.chat.completions.create(
        model=model_mod.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user",   "content": f"Question: {question}\nGround Truth: {gt_str}\nPredicted: {pred_str}"},
        ],
        max_completion_tokens=10, temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip().lower() == "correct"


def stage_apply_and_eval(
    *,
    apply_strategy: str,
    question: str,
    question_slug: str,
    rule_folder: Path,            # the rule set to apply (refined if Stage 3 ran, else Stage 2 pool)
    fallback_folder: Path | None, # full pool, only needed for apply_strategy="default"
    sample_docs: dict[str, dict],
    unsampled_docs: dict[str, dict],
    sample_labels: dict[str, dict],
    unsampled_labels: dict[str, dict],
    rules_dir: Path,
    output_dir: Path,
    model: str,
    skip_existing: bool,
) -> dict[str, dict]:
    """Return {'sampled': eval_dict, 'unsampled': eval_dict}."""

    rule_names = _list_rule_names(rule_folder)
    if not rule_names:
        raise ValueError(f"No rules in {rule_folder}")
    rule_set_slug = "__".join(sorted(rule_names))[:120]

    eval_results: dict[str, dict] = {}
    model_mod = importlib.import_module(f"models.{model}")

    splits = [
        ("sampled",   sample_docs,    sample_labels,    "rule_run"),
        ("unsampled", unsampled_docs, unsampled_labels, "rule_run_unsampled"),
    ]

    for split, doc_map, labels_dict, subdir in splits:
        run_dir  = output_dir / subdir / apply_strategy / question_slug
        run_file = run_dir / f"{rule_set_slug}.json"
        eval_out = output_dir / "eval" / f"{question_slug}_{split}.json"

        already = set()
        existing = _read_json(run_file, default=[])
        if isinstance(existing, list):
            already = {r.get("doc_name", "") for r in existing}

        if skip_existing and already.issuperset(doc_map.keys()) and eval_out.exists():
            print(f"  [apply+eval:{split}] SKIP", flush=True)
            ed = _read_json(eval_out, default={})
            eval_results[split] = {k: v for k, v in ed.items() if k != "per_doc"}
            continue

        # ── Apply ──
        run_dir.mkdir(parents=True, exist_ok=True)
        remaining = {dn: d for dn, d in doc_map.items() if dn not in already}
        print(f"  [apply:{split}] {apply_strategy}  rules={len(rule_names)}  docs={len(remaining)}", flush=True)

        for doc_name, document in remaining.items():
            try:
                _apply_one(
                    apply_strategy   = apply_strategy,
                    document         = document,
                    rule_names       = rule_names,
                    rule_folder      = rule_folder,
                    fallback_folder  = fallback_folder,
                    question         = question,
                    question_slug    = question_slug,
                    rules_dir        = rules_dir,
                    output_dir       = output_dir / subdir / apply_strategy,
                    model            = model,
                )
            except Exception as e:
                print(f"    ERROR apply {doc_name}: {e}", flush=True)

        # ── Evaluate ──
        records = _read_json(run_file, default=[]) or []
        preds   = {r["doc_name"]: r for r in records}
        per_doc: list[dict] = []
        for doc_name, document in doc_map.items():
            pred_rec  = preds.get(doc_name, {})
            predicted = pred_rec.get("predicted_answer")
            gt        = labels_dict.get(doc_name + ".pdf", {}).get(question)
            correct   = _judge(model_mod, question, gt, predicted)
            total_tok = _count_tokens("\n".join(s.get("text", "") for s in document.get("texts", [])))
            ret_tok   = pred_rec.get("retrieved_token_count", 0)
            per_doc.append({
                "doc_name":         doc_name,
                "predicted":        predicted,
                "ground_truth":     gt,
                "correct":          correct,
                "latency_seconds":  pred_rec.get("latency_seconds", 0.0),
                "retrieved_tokens": ret_tok,
                "input_tokens":     pred_rec.get("input_tokens", 0),
                "cost_ratio":       (ret_tok / total_tok) if total_tok > 0 else 0.0,
                "used_fallback":    pred_rec.get("used_fallback"),
            })

        eval_record = {
            "question":        question,
            "question_slug":   question_slug,
            "split":           split,
            "apply_strategy":  apply_strategy,
            "n":               len(per_doc),
            "n_correct":       sum(1 for r in per_doc if r["correct"]),
            "accuracy":        round(mean(int(r["correct"]) for r in per_doc), 4) if per_doc else 0.0,
            "avg_latency":     round(mean(r["latency_seconds"] for r in per_doc), 3) if per_doc else 0.0,
            "avg_retrieved":   round(mean(r["retrieved_tokens"] for r in per_doc), 1) if per_doc else 0.0,
            "avg_input_tok":   round(mean(r["input_tokens"] for r in per_doc), 1) if per_doc else 0.0,
            "avg_cost_ratio":  round(mean(r["cost_ratio"] for r in per_doc), 6) if per_doc else 0.0,
            "per_doc":         per_doc,
        }
        _write_json(eval_out, eval_record)
        eval_results[split] = {k: v for k, v in eval_record.items() if k != "per_doc"}
        print(f"  [eval:{split}] accuracy={eval_record['accuracy']:.3f}  cost_ratio={eval_record['avg_cost_ratio']:.4f}", flush=True)

    return eval_results


def _apply_one(
    *,
    apply_strategy: str,
    document: dict,
    rule_names: list[str],
    rule_folder: Path,
    fallback_folder: Path | None,
    question: str,
    question_slug: str,
    rules_dir: Path,
    output_dir: Path,
    model: str,
):
    if apply_strategy == "merge":
        from rule_apply.merge import rule_apply_merge
        return rule_apply_merge(
            document=document, rule_names=rule_names,
            question_slug=question_slug, question=question,
            rules_dir=str(rule_folder.parent), output_dir=str(output_dir),
            model_name=model,
        )

    if apply_strategy == "default":
        if fallback_folder is None:
            raise ValueError("apply_strategy='default' requires a fallback (Stage 2 pool)")
        from rule_apply.default import apply_with_fallback
        return apply_with_fallback(
            document=document,
            refined_rule_names   = rule_names,
            full_pool_rule_names = _list_rule_names(fallback_folder),
            question=question, question_slug=question_slug,
            rules_dir=str(rules_dir),
        )

    raise ValueError(f"Unknown apply_strategy: {apply_strategy!r}")


# ── Orchestrator ────────────────────────────────────────────────────────────

def run_pipeline(
    *,
    sampling_strategy: str = "random",
    rule_gen_strategy: str = "llm_coarse",
    refine_strategy:   str = "none",
    apply_strategy:    str = "merge",
    queries_file:      str = "data/financebench/sample_queries.txt",
    dataset:           str = "financebench",
    cluster:           str = "single_cluster",
    processing_dir:    str | None = None,
    output_dir:        str = "results/e2e",
    rules_dir:         str | None = None,
    skip_existing:     bool = False,
    model:             str = "gpt54",
) -> dict:
    """Run the four-stage LSF pipeline once, end to end.

    Strategy choices map to:
      sampling_strategy : {"random", "fps"}
      rule_gen_strategy : {"llm_coarse", "agent_langchain", "agent_claude", "agent_codex"}
      refine_strategy   : {"none", "v1", "p_mini", "p_gpt54", "p_proxy", "p_v2", "p_v3", "agentic"}
      apply_strategy    : {"merge", "default"}

    Returns the same dict that is also persisted to {output_dir}/pipeline_summary.json.
    """
    # ── Validate ──
    if sampling_strategy not in SAMPLING_STRATEGIES:
        raise ValueError(f"sampling_strategy must be one of {SAMPLING_STRATEGIES}, got {sampling_strategy!r}")
    if rule_gen_strategy not in RULE_GEN_STRATEGIES:
        raise ValueError(f"rule_gen_strategy must be one of {RULE_GEN_STRATEGIES}, got {rule_gen_strategy!r}")
    if refine_strategy not in REFINE_STRATEGIES:
        raise ValueError(f"refine_strategy must be one of {REFINE_STRATEGIES}, got {refine_strategy!r}")
    if apply_strategy not in APPLY_STRATEGIES:
        raise ValueError(f"apply_strategy must be one of {APPLY_STRATEGIES}, got {apply_strategy!r}")
    if apply_strategy == "default" and refine_strategy == "none":
        raise ValueError("apply_strategy='default' needs a refined subset; pick a refine_strategy != 'none'.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    processing_dir = processing_dir or str(_ROOT / "data" / dataset / "processing")
    rules_root = Path(rules_dir or (_ROOT / "rules" / dataset / "lsf" / cluster / rule_gen_strategy / model))

    print(f"\n=== Pipeline ===", flush=True)
    print(f"  sampling   : {sampling_strategy}", flush=True)
    print(f"  rule_gen   : {rule_gen_strategy}", flush=True)
    print(f"  refine     : {refine_strategy}", flush=True)
    print(f"  apply      : {apply_strategy}", flush=True)
    print(f"  dataset    : {dataset}  cluster : {cluster}", flush=True)
    print(f"  output_dir : {out}\n", flush=True)

    # ── Stage 1 ──
    print("Stage 1 — Sampling", flush=True)
    sample_path, unsampled_path = stage_sampling(
        strategy=sampling_strategy, dataset=dataset, cluster=cluster,
        output_dir=out, skip_existing=skip_existing,
    )
    sample_labels    = json.loads(sample_path.read_text(encoding="utf-8"))
    unsampled_labels = json.loads(unsampled_path.read_text(encoding="utf-8"))
    sample_doc_map    = _load_docs(sample_labels, processing_dir)
    unsampled_doc_map = _load_docs(unsampled_labels, processing_dir)
    sample_docs       = list(sample_doc_map.values())
    sample_doc_names  = list(sample_doc_map.keys())
    print(f"  sampled={len(sample_doc_map)}  unsampled={len(unsampled_doc_map)}", flush=True)

    questions = [l.strip() for l in Path(queries_file).read_text().splitlines() if l.strip()]
    print(f"  questions={len(questions)}\n", flush=True)

    # ── Per-question loop: Stages 2-4 ──
    per_question_summary: list[dict] = []

    for question in questions:
        slug = _make_slug(question)
        question_slug = f"{slug}_{len(sample_doc_map)}"
        print(f"\n--- Question: {question}", flush=True)

        try:
            ground_truth = {
                k.replace(".pdf", ""): v[question]
                for k, v in sample_labels.items() if question in v
            }

            # Stage 2 — Rule Generation
            print("Stage 2 — Rule Generation", flush=True)
            rule_folder_gen, gen_meta = stage_rule_gen(
                strategy=rule_gen_strategy, question=question, question_slug=question_slug,
                sample_docs=sample_docs, sample_doc_names=sample_doc_names,
                ground_truth=ground_truth, rules_dir=rules_root,
                output_dir=out, model=model, skip_existing=skip_existing,
            )
            n_rules_gen = len(_list_rule_names(rule_folder_gen))

            # Stage 3 — Refinement (optional)
            if refine_strategy == "none":
                effective_folder = rule_folder_gen
                fallback_folder  = None
                n_rules_refined  = None
            else:
                print("Stage 3 — Rule Refinement", flush=True)
                effective_folder = stage_refine(
                    strategy=refine_strategy, question=question, question_slug=question_slug,
                    rule_folder=rule_folder_gen, sample_docs=sample_docs,
                    ground_truth=ground_truth, rules_dir=rules_root,
                    output_dir=out, model=model, skip_existing=skip_existing,
                )
                fallback_folder = rule_folder_gen if apply_strategy == "default" else None
                n_rules_refined = len(_list_rule_names(effective_folder))

            # Stage 4 — Apply + Evaluate
            print("Stage 4 — Apply + Evaluate", flush=True)
            eval_results = stage_apply_and_eval(
                apply_strategy=apply_strategy, question=question, question_slug=question_slug,
                rule_folder=effective_folder, fallback_folder=fallback_folder,
                sample_docs=sample_doc_map, unsampled_docs=unsampled_doc_map,
                sample_labels=sample_labels, unsampled_labels=unsampled_labels,
                rules_dir=rules_root, output_dir=out,
                model=model, skip_existing=skip_existing,
            )

            per_question_summary.append({
                "question":                 question,
                "question_slug":            question_slug,
                "num_rules_generated":      n_rules_gen,
                "num_rules_after_refine":   n_rules_refined,
                "sampled_accuracy":         eval_results.get("sampled",   {}).get("accuracy"),
                "unsampled_accuracy":       eval_results.get("unsampled", {}).get("accuracy"),
                "avg_cost_ratio_sampled":   eval_results.get("sampled",   {}).get("avg_cost_ratio"),
                "avg_cost_ratio_unsampled": eval_results.get("unsampled", {}).get("avg_cost_ratio"),
            })

        except Exception as e:
            print(f"  ERROR on {question_slug}: {e}", flush=True)
            per_question_summary.append({
                "question":      question,
                "question_slug": question_slug,
                "error":         str(e),
            })

    # ── Overall summary ──
    sampled_accs   = [q["sampled_accuracy"]   for q in per_question_summary if q.get("sampled_accuracy")   is not None]
    unsampled_accs = [q["unsampled_accuracy"] for q in per_question_summary if q.get("unsampled_accuracy") is not None]
    cost_ratios    = [q["avg_cost_ratio_unsampled"] for q in per_question_summary if q.get("avg_cost_ratio_unsampled") is not None]

    summary = {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "sampling_strategy":  sampling_strategy,
        "rule_gen_strategy":  rule_gen_strategy,
        "refine_strategy":    refine_strategy,
        "apply_strategy":     apply_strategy,
        "queries_file":       queries_file,
        "dataset":            dataset,
        "cluster":            cluster,
        "num_questions":      len(questions),
        "num_sampled_docs":   len(sample_doc_map),
        "num_unsampled_docs": len(unsampled_doc_map),
        "questions":          per_question_summary,
        "overall": {
            "avg_sampled_accuracy":   round(mean(sampled_accs),   4) if sampled_accs   else None,
            "avg_unsampled_accuracy": round(mean(unsampled_accs), 4) if unsampled_accs else None,
            "avg_cost_ratio":         round(mean(cost_ratios),    6) if cost_ratios    else None,
        },
    }
    _write_json(out / "pipeline_summary.json", summary)
    print(f"\n=== Done. Summary -> {out / 'pipeline_summary.json'} ===", flush=True)
    return summary


# ── CLI ─────────────────────────────────────────────────────────────────────

def _build_cli():
    p = argparse.ArgumentParser(
        description="LSF end-to-end pipeline: sampling -> rule_gen -> refine -> apply + eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sampling-strategy", default="random", choices=SAMPLING_STRATEGIES)
    p.add_argument("--rule-gen-strategy", default="llm_coarse", choices=RULE_GEN_STRATEGIES)
    p.add_argument("--refine-strategy",   default="none", choices=REFINE_STRATEGIES)
    p.add_argument("--apply-strategy",    default="merge", choices=APPLY_STRATEGIES)
    p.add_argument("--queries-file",      default="data/financebench/sample_queries.txt")
    p.add_argument("--dataset",           default="financebench")
    p.add_argument("--cluster",           default="single_cluster")
    p.add_argument("--processing-dir",    default=None)
    p.add_argument("--rules-dir",         default=None)
    p.add_argument("--output-dir",        default="results/e2e")
    p.add_argument("--model",             default="gpt54")
    p.add_argument("--skip-existing",     action="store_true")
    return p


def main():
    args = _build_cli().parse_args()
    run_pipeline(
        sampling_strategy = args.sampling_strategy,
        rule_gen_strategy = args.rule_gen_strategy,
        refine_strategy   = args.refine_strategy,
        apply_strategy    = args.apply_strategy,
        queries_file      = args.queries_file,
        dataset           = args.dataset,
        cluster           = args.cluster,
        processing_dir    = args.processing_dir,
        rules_dir         = args.rules_dir,
        output_dir        = args.output_dir,
        model             = args.model,
        skip_existing     = args.skip_existing,
    )


if __name__ == "__main__":
    main()
