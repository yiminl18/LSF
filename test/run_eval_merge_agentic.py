"""Evaluate the agentic-generation rule pool on sampled / unsampled docs.

Mirrors test/run_eval_merge_{sampled,unsampled}.py but parameterised on which
sample set the rules were generated from (random vs FPS) and which split to
evaluate against (sampled vs unsampled).

Outputs go to <results_dir>/eval_merge_{sampled,unsampled}/<slug>.json,
where <results_dir> = results/financebench_single_cluster/agent/opus47/
                       {agentic | agentic_fps}/raw/

Usage:
    # Task 1 evals: rules generated from random sample
    python test/run_eval_merge_agentic.py --sample-set random --split sampled
    python test/run_eval_merge_agentic.py --sample-set random --split unsampled

    # Task 2 evals: rules generated from FPS sample
    python test/run_eval_merge_agentic.py --sample-set fps --split sampled
    python test/run_eval_merge_agentic.py --sample-set fps --split unsampled
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import warnings
from pathlib import Path
from statistics import mean

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

from rule_apply_merge import rule_apply_merge       # noqa: E402

model_mod = importlib.import_module("models.gpt54")

QUERIES_FILE   = "data/financebench/sample_queries.txt"
PROCESSING_DIR = "data/financebench/processing"


# ── Sample-set config ────────────────────────────────────────────────────────

def sample_set_config(name: str) -> dict:
    if name == "random":
        return {
            "sampled_labels":   "data/financebench/sample_doc_labels.json",
            "unsampled_labels": "data/financebench/unsampled_doc_labels.json",
            "slug_suffix":      "_10_agentic",
            "rules_dir":        "rules/financebench_single_cluster/agent/opus47/agentic/raw",
            "results_dir":      "results/financebench_single_cluster/agent/opus47/agentic/raw",
        }
    if name == "fps":
        return {
            "sampled_labels":   "data/financebench/sample/single_cluster/fps/sample_doc_labels.json",
            "unsampled_labels": "data/financebench/sample/single_cluster/fps/unsampled_doc_labels.json",
            "slug_suffix":      "_10_agentic_fps",
            "rules_dir":        "rules/financebench_single_cluster/agent/opus47/agentic_fps/raw",
            "results_dir":      "results/financebench_single_cluster/agent/opus47/agentic_fps/raw",
        }
    raise ValueError(f"unknown sample-set: {name!r}")


# ── Judge ────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
You are an answer equivalence judge for a financial document QA system.
You will be given a question, a predicted answer, and a ground truth answer.
Judge whether the predicted answer is correct — meaning semantically equivalent
to the ground truth, ignoring minor formatting differences.

Equivalence rules:
- Treat "2017" and "year 2017" as the same
- Treat "$4.5 billion" and "4,500 million" as the same if numerically equal
- Treat "NYSE" and "New York Stock Exchange" as the same
- Ignore leading/trailing whitespace, punctuation, and capitalization differences
- If the predicted answer is "NOT FOUND" or null, always judge as incorrect

Reply with exactly one word: CORRECT or INCORRECT"""


def judge(question: str, ground_truth, predicted) -> bool:
    if ground_truth is None or predicted is None:
        return False
    gt_str   = json.dumps(ground_truth) if not isinstance(ground_truth, str) else ground_truth
    pred_str = str(predicted)
    user_msg = f"Question: {question}\nGround Truth: {gt_str}\nPredicted: {pred_str}"
    response = model_mod.client.chat.completions.create(
        model=model_mod.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        max_completion_tokens=10,
        temperature=0.0,
    )
    verdict = (response.choices[0].message.content or "").strip().lower()
    if verdict not in ("correct", "incorrect"):
        warnings.warn(f"Unexpected judge response: '{verdict}'")
    return verdict == "correct"


def count_tokens(text: str) -> int:
    return len(text.split())


def make_slug(q: str, suffix: str) -> str:
    s = q.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60] + suffix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-set", choices=("random", "fps"), required=True)
    ap.add_argument("--split",      choices=("sampled", "unsampled"), required=True)
    args = ap.parse_args()

    cfg = sample_set_config(args.sample_set)
    labels_file = cfg["sampled_labels"] if args.split == "sampled" else cfg["unsampled_labels"]
    rules_base  = cfg["rules_dir"]
    results_dir = cfg["results_dir"]
    out_subdir  = "eval_merge_sampled" if args.split == "sampled" else "eval_merge_unsampled"
    merge_run_dir = f"{results_dir}/rule_run_merge_{args.split}"
    output_dir    = f"{results_dir}/{out_subdir}"

    questions = [l.strip() for l in open(QUERIES_FILE) if l.strip()]
    labels: dict[str, dict] = json.loads(Path(labels_file).read_text(encoding="utf-8"))

    doc_map: dict[str, dict] = {}
    for pdf_key in labels:
        doc_name = pdf_key.replace(".pdf", "")
        path = Path(PROCESSING_DIR) / f"{doc_name}_reconstructed.json"
        if path.exists():
            doc_map[doc_name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            print(f"WARNING: missing {path}")
    doc_total_tokens = {
        doc_name: count_tokens("\n".join(s.get("text", "") for s in doc.get("texts", [])))
        for doc_name, doc in doc_map.items()
    }

    Path(merge_run_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"sample_set={args.sample_set}  split={args.split}")
    print(f"questions={len(questions)}  docs={len(doc_map)}  rules_base={rules_base}\n")

    summary: list[dict] = []
    for question in questions:
        rule_slug = make_slug(question, cfg["slug_suffix"])
        rule_folder = Path(rules_base) / rule_slug
        out_file    = Path(output_dir) / f"{rule_slug}_{args.split}.json"

        if out_file.exists():
            print(f"SKIP (exists): {out_file.name}")
            continue
        if not rule_folder.is_dir():
            print(f"SKIP (no rule folder): {rule_folder}")
            continue
        rule_names = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(rule_folder)
            if f.startswith("rule_") and f.endswith(".py")
        )
        if not rule_names:
            print(f"SKIP (empty): {rule_folder}")
            continue

        print(f"\nQuestion: {question}\n  rules={len(rule_names)}  docs={len(doc_map)}")

        run_results: list[dict] = []
        for doc_name, document in doc_map.items():
            try:
                result = rule_apply_merge(
                    document=document,
                    rule_names=rule_names,
                    question_slug=rule_slug,
                    question=question,
                    rules_dir=rules_base,
                    output_dir=merge_run_dir,
                )
                run_results.append({
                    "doc_name":         doc_name,
                    "predicted":        result["predicted_answer"],
                    "latency_seconds":  result["latency_seconds"],
                    "retrieved_tokens": result["retrieved_token_count"],
                    "input_tokens":     result["input_tokens"],
                })
            except Exception as e:
                print(f"  ERROR {doc_name}: {e}")
                run_results.append({"doc_name": doc_name, "predicted": None,
                                    "latency_seconds": 0.0, "retrieved_tokens": 0, "input_tokens": 0})

        per_doc_out: list[dict] = []
        cost_ratios: list[float] = []
        for r in run_results:
            doc_name     = r["doc_name"]
            ground_truth = labels.get(doc_name + ".pdf", {}).get(question)
            correct      = judge(question, ground_truth, r["predicted"])
            total_tok    = doc_total_tokens.get(doc_name, 1)
            cost_ratio   = r["retrieved_tokens"] / total_tok if total_tok > 0 else 0.0
            cost_ratios.append(cost_ratio)
            per_doc_out.append({
                "doc_name":         doc_name,
                "predicted":        r["predicted"],
                "ground_truth":     ground_truth,
                "correct":          correct,
                "latency_seconds":  r["latency_seconds"],
                "retrieved_tokens": r["retrieved_tokens"],
                "input_tokens":     r["input_tokens"],
            })

        n              = len(per_doc_out)
        n_correct      = sum(r["correct"] for r in per_doc_out)
        accuracy       = round(n_correct / n, 4) if n else 0.0
        avg_latency    = round(mean(r["latency_seconds"]  for r in per_doc_out), 3)
        avg_retrieved  = round(mean(r["retrieved_tokens"] for r in per_doc_out), 1)
        avg_input_tok  = round(mean(r["input_tokens"]     for r in per_doc_out), 1)
        avg_cost_ratio = round(mean(cost_ratios), 6)

        print(f"  accuracy={accuracy:.2f} ({n_correct}/{n})  cost={avg_cost_ratio:.5f}")

        per_q = {
            "question":       question,
            "question_slug":  rule_slug,
            "split":          args.split,
            "n":              n,
            "n_correct":      n_correct,
            "accuracy":       accuracy,
            "avg_latency":    avg_latency,
            "avg_retrieved":  avg_retrieved,
            "avg_input_tok":  avg_input_tok,
            "avg_cost_ratio": avg_cost_ratio,
            "n_rules":        len(rule_names),
            "per_doc":        per_doc_out,
        }
        out_file.write_text(json.dumps(per_q, indent=2, ensure_ascii=False), encoding="utf-8")
        summary.append(per_q)

    # Write/update summary
    summary_path = Path(output_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if summary:
        avg_acc  = round(mean(s["accuracy"]       for s in summary), 4)
        avg_cost = round(mean(s["avg_cost_ratio"] for s in summary), 6)
        print(f"\n{'='*72}")
        print(f"  mean accuracy:   {avg_acc}")
        print(f"  mean cost ratio: {avg_cost}")
        print(f"  written to:      {output_dir}/")


if __name__ == "__main__":
    main()
