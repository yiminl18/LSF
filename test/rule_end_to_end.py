"""End-to-end rule pipeline: generation → (optional) refinement → application → evaluation."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import signal
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

from rule_apply_merge import rule_apply_merge


# ── Timeout helper (SIGALRM, Linux/macOS only) ─────────────────────────────────

class _RuleGenTimeout(Exception):
    pass

def _run_with_timeout(fn, timeout_secs, **kwargs):
    if timeout_secs <= 0:
        return fn(**kwargs)
    def _handler(signum, frame):
        raise _RuleGenTimeout(f"rule gen timed out after {timeout_secs}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_secs)
    try:
        return fn(**kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ── Token counting ─────────────────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


# ── Judge ──────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are an answer equivalence judge for a financial document QA system.\n"
    "You will be given a question, a predicted answer, and a ground truth answer.\n"
    "Judge whether the predicted answer is correct — meaning semantically equivalent\n"
    "to the ground truth, ignoring minor formatting differences.\n\n"
    "Equivalence rules:\n"
    '- Treat "2017" and "year 2017" as the same\n'
    '- Treat "$4.5 billion" and "4,500 million" as the same if numerically equal\n'
    '- Treat "NYSE" and "New York Stock Exchange" as the same\n'
    "- Ignore leading/trailing whitespace, punctuation, and capitalization differences\n"
    '- If the predicted answer is "NOT FOUND" or null, always judge as incorrect\n\n'
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
        max_completion_tokens=10,
        temperature=0.0,
    )
    verdict = (resp.choices[0].message.content or "").strip().lower()
    if verdict not in ("correct", "incorrect"):
        warnings.warn(f"Unexpected judge verdict: '{verdict}'")
    return verdict == "correct"


# ── Slug ───────────────────────────────────────────────────────────────────────

def _make_slug(q: str) -> str:
    s = q.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60]


def _make_slug_agent(q: str) -> str:
    return re.sub(r"[^\w]", "_", q.lower())[:60].rstrip("_")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_rule_names(folder: Path) -> list[str]:
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(folder)
        if f.startswith("rule_") and f.endswith(".py")
    )


def _load_docs(labels: dict, processing_dir: str) -> dict[str, dict]:
    doc_map: dict[str, dict] = {}
    for pdf_key in labels:
        doc_name = pdf_key.replace(".pdf", "")
        path = Path(processing_dir) / f"{doc_name}_reconstructed.json"
        if path.exists():
            doc_map[doc_name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            print(f"  WARNING: missing {path}", flush=True)
    return doc_map


def _update_summary(summary_path: Path, question_slug: str, question: str, splits: dict):
    summary: list[dict] = []
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    entry = {"question": question, "question_slug": question_slug, **splits}
    matched = False
    for i, e in enumerate(summary):
        if e.get("question_slug") == question_slug:
            summary[i] = entry
            matched = True
            break
    if not matched:
        summary.append(entry)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="End-to-end rule pipeline")
    parser.add_argument("--rule-gen-module",  default="src/rule_gen_llm_coarse.py")
    parser.add_argument("--queries-file",     default="data/financebench/sample_queries.txt")
    parser.add_argument("--sample-labels",    default="data/financebench/sample_doc_labels.json")
    parser.add_argument("--unsampled-labels", default="data/financebench/unsampled_doc_labels.json")
    parser.add_argument("--processing-dir",   default="data/financebench/processing")
    parser.add_argument("--rules-dir",        default="rules/llm/financebench")
    parser.add_argument("--output-dir",       default="results/e2e")
    parser.add_argument("--use-refine",        action="store_true")
    parser.add_argument("--skip-existing",     action="store_true")
    parser.add_argument("--agent-rules",       action="store_true",
                        help="Skip rule gen; load rules from agent-style slug folders (no _llm suffix).")
    parser.add_argument("--rule-gen-timeout",  type=int, default=0,
                        help="Timeout in seconds for rule gen per query (0 = no limit)")
    args = parser.parse_args()

    # ── Dynamic import of rule_gen function ───────────────────────────────────
    spec = importlib.util.spec_from_file_location("rule_gen_mod", args.rule_gen_module)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rule_gen_fn = next(v for k, v in vars(mod).items() if k.startswith("rule_gen_") and callable(v))

    model_mod = importlib.import_module("models.gpt54")

    # ── Load data ─────────────────────────────────────────────────────────────
    questions = [l.strip() for l in open(args.queries_file) if l.strip()]
    sample_labels:    dict[str, dict] = json.loads(Path(args.sample_labels).read_text(encoding="utf-8"))
    unsampled_labels: dict[str, dict] = json.loads(Path(args.unsampled_labels).read_text(encoding="utf-8"))

    print("Loading sampled docs ...", flush=True)
    sample_doc_map    = _load_docs(sample_labels,    args.processing_dir)
    print("Loading unsampled docs ...", flush=True)
    unsampled_doc_map = _load_docs(unsampled_labels, args.processing_dir)
    sample_docs       = list(sample_doc_map.values())
    print(f"Questions: {len(questions)}  sampled: {len(sample_doc_map)}  unsampled: {len(unsampled_doc_map)}\n", flush=True)

    # ── Create output directories ─────────────────────────────────────────────
    out = Path(args.output_dir)
    for d in ["rule_gen", "rule_run/merge", "rule_run_unsampled/merge", "eval"]:
        (out / d).mkdir(parents=True, exist_ok=True)
    if args.use_refine:
        (out / "refined_rules").mkdir(parents=True, exist_ok=True)
        (out / "rule_refine").mkdir(parents=True, exist_ok=True)

    pipeline_questions: list[dict] = []

    n_sample_docs = len(sample_doc_map)
    for question in questions:
        slug          = _make_slug(question)
        question_slug = f"{slug}_{n_sample_docs}"
        print(f"\nQuestion: {question}", flush=True)

        try:
            # ── Stage 1: Rule Generation ───────────────────────────────────────
            if args.agent_rules:
                agent_slug      = _make_slug_agent(question)
                rule_folder_gen = Path(args.rules_dir) / agent_slug
                print(f"  [gen] SKIP (--agent-rules) using folder: {agent_slug}", flush=True)
            else:
                rule_folder_gen = Path(args.rules_dir) / f"{question_slug}_llm"
                rule_gen_out    = out / "rule_gen" / f"{question_slug}_rule_gen.json"

                if args.skip_existing and rule_folder_gen.is_dir():
                    print(f"  [gen] SKIP {question_slug}", flush=True)
                else:
                    ground_truth = {
                        k: sample_labels[k][question]
                        for k in sample_labels if question in sample_labels[k]
                    }
                    gen_result = _run_with_timeout(
                        rule_gen_fn,
                        args.rule_gen_timeout,
                        documents=sample_docs,
                        question=question,
                        ground_truth=ground_truth,
                        rules_dir=args.rules_dir,
                        output_dir=str(out / "rule_gen"),
                    )
                    rule_gen_out.write_text(json.dumps(gen_result, indent=2, ensure_ascii=False), encoding="utf-8")
                    print(f"  [gen] {question_slug}: {len(gen_result['rules'])} rules", flush=True)

            if not rule_folder_gen.is_dir():
                print(f"  WARNING: no rule folder {rule_folder_gen}, skipping", flush=True)
                continue

            rule_names_gen    = _load_rule_names(rule_folder_gen)
            num_rules_gen     = len(rule_names_gen)
            num_rules_refined = None

            # ── Stage 2: Rule Refinement (optional) ───────────────────────────
            if args.use_refine:
                from rule_refine import rule_refine, evaluate_merge_accuracy

                refine_out_path  = out / "rule_refine" / f"{question_slug}_refine.json"
                refined_rules_dir = str(out / "refined_rules")

                if args.skip_existing and refine_out_path.exists():
                    print(f"  [refine] SKIP {question_slug}", flush=True)
                    rd = json.loads(refine_out_path.read_text(encoding="utf-8"))
                    num_rules_refined = rd["selected_rules_count"]
                else:
                    ground_truth = {
                        k: sample_labels[k][question]
                        for k in sample_labels if question in sample_labels[k]
                    }
                    # compute target accuracy inline using all generated rules
                    target_accuracy, *_ = evaluate_merge_accuracy(
                        rule_names=rule_names_gen,
                        documents=sample_docs,
                        ground_truth=ground_truth,
                        question=question,
                        rule_folder=rule_folder_gen,
                        model_mod=model_mod,
                    )
                    print(f"  [refine] target_accuracy={target_accuracy:.2f}", flush=True)

                    refine_result = rule_refine(
                        rule_names=rule_names_gen,
                        target_accuracy=target_accuracy,
                        question=question,
                        question_slug=question_slug,
                        documents=sample_docs,
                        ground_truth=ground_truth,
                        rules_dir=args.rules_dir,
                        output_dir=refined_rules_dir,
                    )
                    refine_out_path.write_text(json.dumps(refine_result, indent=2, ensure_ascii=False), encoding="utf-8")
                    num_rules_refined = refine_result["selected_rules_count"]
                    print(
                        f"  [refine] {question_slug}: {num_rules_refined} rules kept  "
                        f"cost_reduction={refine_result['cost_reduction_ratio']:.1%}",
                        flush=True,
                    )

                effective_rules_dir = refined_rules_dir
                apply_slug          = question_slug            # {slug}_10
            elif args.agent_rules:
                effective_rules_dir = args.rules_dir
                apply_slug          = _make_slug_agent(question)
            else:
                effective_rules_dir = args.rules_dir
                apply_slug          = f"{question_slug}_llm"  # {slug}_10_llm

            rule_folder_eff = Path(effective_rules_dir) / apply_slug
            if not rule_folder_eff.is_dir():
                print(f"  WARNING: no effective rule folder {rule_folder_eff}, skipping", flush=True)
                continue

            rule_names = _load_rule_names(rule_folder_eff)
            if not rule_names:
                print(f"  WARNING: empty rule folder {rule_folder_eff}, skipping", flush=True)
                continue

            rule_set_slug = "__".join(sorted(rule_names))[:120]

            # ── Stage 3: Rule Application ──────────────────────────────────────
            for split, labels_dict, doc_map, run_subdir in [
                ("sampled",   sample_labels,   sample_doc_map,   "rule_run/merge"),
                ("unsampled", unsampled_labels, unsampled_doc_map, "rule_run_unsampled/merge"),
            ]:
                run_out_dir = str(out / run_subdir)
                run_file    = out / run_subdir / apply_slug / f"{rule_set_slug}_merge.json"

                already_done: set[str] = set()
                if run_file.exists():
                    try:
                        existing = json.loads(run_file.read_text(encoding="utf-8"))
                        already_done = {r.get("doc_name", "") for r in existing}
                    except Exception:
                        pass

                if args.skip_existing and already_done.issuperset(doc_map.keys()):
                    print(f"  [apply:{split}] SKIP (all {len(doc_map)} cached)", flush=True)
                else:
                    remaining = {dn: d for dn, d in doc_map.items() if dn not in already_done}
                    print(f"  [apply:{split}] rules={len(rule_names)}  docs={len(remaining)}", flush=True)
                    for doc_name, document in remaining.items():
                        try:
                            rule_apply_merge(
                                document=document,
                                rule_names=rule_names,
                                question_slug=apply_slug,
                                question=question,
                                rules_dir=effective_rules_dir,
                                output_dir=run_out_dir,
                            )
                        except Exception as e:
                            print(f"    ERROR apply {doc_name}: {e}", flush=True)

            # ── Stage 4: Evaluation ────────────────────────────────────────────
            eval_splits: dict[str, dict] = {}

            for split, labels_dict, doc_map, run_subdir in [
                ("sampled",   sample_labels,   sample_doc_map,   "rule_run/merge"),
                ("unsampled", unsampled_labels, unsampled_doc_map, "rule_run_unsampled/merge"),
            ]:
                eval_out = out / "eval" / f"{question_slug}_{split}.json"

                if args.skip_existing and eval_out.exists():
                    print(f"  [eval:{split}] SKIP", flush=True)
                    ed = json.loads(eval_out.read_text(encoding="utf-8"))
                    eval_splits[split] = {k: v for k, v in ed.items() if k != "per_doc"}
                    continue

                run_file = out / run_subdir / apply_slug / f"{rule_set_slug}_merge.json"
                if not run_file.exists():
                    print(f"  WARNING: no run file {run_file}", flush=True)
                    continue

                records     = json.loads(run_file.read_text(encoding="utf-8"))
                predictions = {r["doc_name"]: r for r in records}

                per_doc: list[dict] = []
                for doc_name, document in doc_map.items():
                    pred_rec        = predictions.get(doc_name, {})
                    predicted       = pred_rec.get("predicted_answer")
                    gt              = labels_dict.get(doc_name + ".pdf", {}).get(question)
                    correct         = _judge(model_mod, question, gt, predicted)
                    total_tok       = _count_tokens("\n".join(s.get("text", "") for s in document.get("texts", [])))
                    retrieved_tokens = pred_rec.get("retrieved_token_count", 0)
                    cost_ratio      = retrieved_tokens / total_tok if total_tok > 0 else 0.0
                    per_doc.append({
                        "doc_name":        doc_name,
                        "predicted":       predicted,
                        "ground_truth":    gt,
                        "correct":         correct,
                        "latency_seconds": pred_rec.get("latency_seconds", 0.0),
                        "retrieved_tokens": retrieved_tokens,
                        "input_tokens":    pred_rec.get("input_tokens", 0),
                    })

                n           = len(per_doc)
                n_correct   = sum(r["correct"] for r in per_doc)
                accuracy    = round(n_correct / n, 4) if n else 0.0
                avg_latency    = round(mean(r["latency_seconds"]  for r in per_doc), 3) if per_doc else 0.0
                avg_retrieved  = round(mean(r["retrieved_tokens"] for r in per_doc), 1) if per_doc else 0.0
                avg_input_tok  = round(mean(r["input_tokens"]     for r in per_doc), 1) if per_doc else 0.0

                cost_ratios: list[float] = []
                for r in per_doc:
                    doc_texts  = doc_map[r["doc_name"]].get("texts", [])
                    total_tok  = _count_tokens("\n".join(s.get("text", "") for s in doc_texts))
                    cost_ratios.append(r["retrieved_tokens"] / total_tok if total_tok > 0 else 0.0)
                avg_cost_ratio = round(mean(cost_ratios), 6) if cost_ratios else 0.0

                eval_data = {
                    "question":       question,
                    "question_slug":  question_slug,
                    "split":          split,
                    "n":              n,
                    "n_correct":      n_correct,
                    "accuracy":       accuracy,
                    "avg_latency":    avg_latency,
                    "avg_retrieved":  avg_retrieved,
                    "avg_input_tok":  avg_input_tok,
                    "avg_cost_ratio": avg_cost_ratio,
                    "per_doc":        per_doc,
                }
                eval_out.write_text(json.dumps(eval_data, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"  [eval:{split}] accuracy={accuracy:.2f} ({n_correct}/{n})  cost={avg_cost_ratio:.4f}", flush=True)
                eval_splits[split] = {k: v for k, v in eval_data.items() if k != "per_doc"}

            # update eval/summary.json
            _update_summary(
                out / "eval" / "summary.json",
                question_slug, question,
                {k: eval_splits[k] for k in eval_splits},
            )

            pipeline_questions.append({
                "question":                question,
                "question_slug":           question_slug,
                "num_rules_generated":     num_rules_gen,
                "num_rules_after_refine":  num_rules_refined,
                "sampled_accuracy":        eval_splits.get("sampled",   {}).get("accuracy"),
                "unsampled_accuracy":      eval_splits.get("unsampled", {}).get("accuracy"),
                "avg_cost_ratio_sampled":  eval_splits.get("sampled",   {}).get("avg_cost_ratio"),
                "avg_cost_ratio_unsampled":eval_splits.get("unsampled", {}).get("avg_cost_ratio"),
            })

        except _RuleGenTimeout as e:
            print(f"  TIMEOUT for '{question}': {e}", flush=True)
            continue
        except Exception as e:
            import traceback
            print(f"  FATAL ERROR for '{question}': {e}", flush=True)
            traceback.print_exc()
            continue

    # ── Pipeline Summary ───────────────────────────────────────────────────────
    sampled_accs   = [q["sampled_accuracy"]       for q in pipeline_questions if q["sampled_accuracy"]       is not None]
    unsampled_accs = [q["unsampled_accuracy"]      for q in pipeline_questions if q["unsampled_accuracy"]      is not None]
    all_costs      = [q["avg_cost_ratio_sampled"]  for q in pipeline_questions if q["avg_cost_ratio_sampled"]  is not None]

    pipeline_summary = {
        "timestamp":          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rule_gen_module":    args.rule_gen_module,
        "use_refine":         args.use_refine,
        "queries_file":       args.queries_file,
        "num_questions":      len(questions),
        "num_sampled_docs":   len(sample_doc_map),
        "num_unsampled_docs": len(unsampled_doc_map),
        "questions":          pipeline_questions,
        "overall": {
            "avg_sampled_accuracy":   round(mean(sampled_accs),   4) if sampled_accs   else None,
            "avg_unsampled_accuracy": round(mean(unsampled_accs), 4) if unsampled_accs else None,
            "avg_cost_ratio":         round(mean(all_costs),      6) if all_costs      else None,
        },
    }
    (out / "pipeline_summary.json").write_text(
        json.dumps(pipeline_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── Final Table ────────────────────────────────────────────────────────────
    print(f"\n{'=== End-to-End Pipeline Summary ==='}")
    print(f"Rule gen: {args.rule_gen_module}  |  refine: {'on' if args.use_refine else 'off'}")
    hdr = f"{'Question':<50}  {'Rules':>7}  {'SampAcc':>7}  {'UnsmpAcc':>8}  {'CostRatio':>9}"
    print(hdr)
    print("-" * len(hdr))
    for q in pipeline_questions:
        if q["num_rules_after_refine"] is not None:
            rules_str = f"{q['num_rules_after_refine']}/{q['num_rules_generated']}"
        else:
            rules_str = str(q["num_rules_generated"])
        print(
            f"  {q['question'][:48]:<48}  {rules_str:>7}  "
            f"{(q['sampled_accuracy']      or 0):>7.2f}  "
            f"{(q['unsampled_accuracy']    or 0):>8.2f}  "
            f"{(q['avg_cost_ratio_sampled'] or 0):>9.4f}",
            flush=True,
        )
    print("-" * len(hdr))
    if sampled_accs:
        print(
            f"  {'AVERAGE':<48}  {'':>7}  "
            f"{mean(sampled_accs):>7.2f}  "
            f"{mean(unsampled_accs) if unsampled_accs else 0:>8.2f}  "
            f"{mean(all_costs) if all_costs else 0:>9.4f}",
            flush=True,
        )
    print(f"\nResults: {args.output_dir}/")

    # ── Auto git push ─────────────────────────────────────────────────────────
    import subprocess
    try:
        result_dir = str(out)
        rules_dir  = args.rules_dir
        subprocess.run(["git", "add", result_dir, rules_dir], check=True)
        subprocess.run(["git", "commit", "-m", f"pipeline results: {Path(result_dir).name}"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("  [git] pushed results to remote", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"  [git] push failed: {e}", flush=True)


if __name__ == "__main__":
    main()
