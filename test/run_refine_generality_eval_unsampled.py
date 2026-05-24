"""Evaluate refine_generality selected rules on unsampled docs for every question.

Reads selected rule sets from rules/.../refine_generality/<slug>_10_refgen/,
applies them in merge mode on the held-out unsampled document set, and
judges predictions against unsampled ground truth.

Prerequisites:
    test/run_refine_generality_select.py must have been run first.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from statistics import mean

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

from rule_apply_merge import rule_apply_merge
from rule_refinement.eval_judge import judge

QUERIES_FILE    = "data/financebench/sample_queries.txt"
LABELS_FILE     = "data/financebench/sample/single_cluster/random/unsampled_doc_labels.json"
PROCESSING_DIR  = "data/financebench/processing"
RULES_BASE_DIR  = "rules/financebench/lsf/single_cluster/llm/gpt54/refine_generality"
MERGE_RUN_DIR   = "results/financebench/lsf/single_cluster/llm/gpt54/refine_generality/rule_run_merge_unsampled"
OUTPUT_DIR      = "results/financebench/lsf/single_cluster/llm/gpt54/refine_generality/eval_merge"
MODEL_NAME      = "gpt54"


def make_slug(q: str) -> str:
    s = q.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60]


def count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


# ── Load shared data ───────────────────────────────────────────────────────────
questions = [l.strip() for l in open(QUERIES_FILE) if l.strip()]
labels: dict = json.loads(Path(LABELS_FILE).read_text(encoding="utf-8"))

doc_map: dict[str, dict] = {}
for pdf_key in labels:
    doc_name = pdf_key.replace(".pdf", "")
    path = Path(PROCESSING_DIR) / f"{doc_name}_reconstructed.json"
    if path.exists():
        doc_map[doc_name] = json.loads(path.read_text(encoding="utf-8"))
    else:
        print(f"WARNING: missing {path}")

print(f"Questions: {len(questions)}  |  Docs: {len(doc_map)}\n")

doc_total_tokens: dict[str, int] = {
    doc_name: count_tokens("\n".join(s.get("text", "") for s in doc.get("texts", [])))
    for doc_name, doc in doc_map.items()
}

Path(MERGE_RUN_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

summary_updates: list[dict] = []

# ── Per-question eval ──────────────────────────────────────────────────────────
for question in questions:
    slug        = make_slug(question)
    rule_slug   = f"{slug}_10_refgen"
    out_slug    = f"{slug}_10"
    rule_folder = Path(RULES_BASE_DIR) / rule_slug
    out_file    = Path(OUTPUT_DIR) / f"{out_slug}_unsampled.json"

    if out_file.exists():
        print(f"SKIP (exists): {out_file.name}")
        existing = json.loads(out_file.read_text(encoding="utf-8"))
        summary_updates.append({
            "question": existing["question"],
            "question_slug": existing["question_slug"],
            "unsampled": {k: v for k, v in existing.items() if k not in ("per_doc", "question", "question_slug")},
        })
        continue

    if not rule_folder.is_dir():
        print(f"SKIP (no rule folder — run select first): {rule_folder}")
        continue

    rule_names = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(rule_folder)
        if f.startswith("rule_") and f.endswith(".py")
    )
    if not rule_names:
        print(f"SKIP (empty rule folder): {rule_folder}")
        continue

    print(f"\nQuestion: {question}  ({len(rule_names)} rules  |  {len(doc_map)} docs)")

    try:
        run_results: list[dict] = []
        q_start = time.time()

        for doc_name, document in doc_map.items():
            try:
                result = rule_apply_merge(
                    document=document,
                    rule_names=rule_names,
                    question_slug=rule_slug,
                    question=question,
                    rules_dir=RULES_BASE_DIR,
                    output_dir=MERGE_RUN_DIR,
                )
                run_results.append({
                    "doc_name":         doc_name,
                    "predicted":        result["predicted_answer"],
                    "latency_seconds":  result["latency_seconds"],
                    "retrieved_tokens": result["retrieved_token_count"],
                    "input_tokens":     result["input_tokens"],
                })
                print(f"  {doc_name}: {str(result['predicted_answer'])[:50]}")
            except Exception as e:
                print(f"  ERROR {doc_name}: {e}")
                run_results.append({
                    "doc_name":         doc_name,
                    "predicted":        None,
                    "latency_seconds":  0.0,
                    "retrieved_tokens": 0,
                    "input_tokens":     0,
                })

        # Judge predictions
        per_doc_out: list[dict] = []
        cost_ratios: list[float] = []

        for r in run_results:
            doc_name     = r["doc_name"]
            ground_truth = labels.get(doc_name + ".pdf", {}).get(question)
            correct, _, _ = judge(question, ground_truth, r["predicted"], model_name=MODEL_NAME)
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

        q_latency = round(time.time() - q_start, 2)
        n              = len(per_doc_out)
        n_correct      = sum(r["correct"] for r in per_doc_out)
        accuracy       = round(n_correct / n, 4) if n else 0.0
        avg_latency    = round(mean(r["latency_seconds"]  for r in per_doc_out), 3)
        avg_retrieved  = round(mean(r["retrieved_tokens"] for r in per_doc_out), 1)
        avg_input_tok  = round(mean(r["input_tokens"]     for r in per_doc_out), 1)
        avg_cost_ratio = round(mean(cost_ratios), 6)

        print(
            f"  accuracy={accuracy:.2f} ({n_correct}/{n})  "
            f"avg_cost_ratio={avg_cost_ratio:.5f}  "
            f"avg_latency={avg_latency:.3f}s  "
            f"total_latency={q_latency:.1f}s"
        )

        per_q = {
            "question":             question,
            "question_slug":        out_slug,
            "rule_slug":            rule_slug,
            "selected_rules":       rule_names,
            "split":                "unsampled",
            "n":                    n,
            "n_correct":            n_correct,
            "accuracy":             accuracy,
            "avg_latency":          avg_latency,
            "avg_retrieved":        avg_retrieved,
            "avg_input_tok":        avg_input_tok,
            "avg_cost_ratio":       avg_cost_ratio,
            "total_eval_latency_seconds": q_latency,
            "per_doc":              per_doc_out,
        }
        out_file.write_text(json.dumps(per_q, indent=2, ensure_ascii=False), encoding="utf-8")

        summary_updates.append({
            "question":      question,
            "question_slug": out_slug,
            "unsampled": {
                "n": n, "n_correct": n_correct, "accuracy": accuracy,
                "avg_latency": avg_latency, "avg_retrieved": avg_retrieved,
                "avg_input_tok": avg_input_tok, "avg_cost_ratio": avg_cost_ratio,
                "total_eval_latency_seconds": q_latency,
            },
        })

    except Exception as e:
        import traceback
        print(f"  FATAL ERROR for '{question}': {e}")
        traceback.print_exc()
        continue

# ── Merge into summary.json ────────────────────────────────────────────────────
summary_path = Path(OUTPUT_DIR) / "summary.json"
existing_summary: list[dict] = []
if summary_path.exists():
    try:
        existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        pass

for new_entry in summary_updates:
    qs = new_entry["question_slug"]
    matched = False
    for entry in existing_summary:
        if entry.get("question_slug") == qs:
            entry["unsampled"] = new_entry["unsampled"]
            matched = True
            break
    if not matched:
        existing_summary.append({"question": new_entry["question"], "question_slug": qs, "unsampled": new_entry["unsampled"]})

summary_path.write_text(json.dumps(existing_summary, indent=2, ensure_ascii=False), encoding="utf-8")

# ── Final table ────────────────────────────────────────────────────────────────
print(f"\n{'Question':<55}  {'Acc':>5}  {'AvgTok':>7}  {'CostRatio':>10}  {'Lat(s)':>7}")
print("-" * 90)
for s in summary_updates:
    u = s["unsampled"]
    print(
        f"  {s['question'][:53]:<53}  {u['accuracy']:>5.2f}  "
        f"{u['avg_retrieved']:>7.1f}  {u['avg_cost_ratio']:>10.5f}  "
        f"{u['total_eval_latency_seconds']:>7.1f}"
    )

print(f"\nResults : {OUTPUT_DIR}/")
print(f"Summary : {summary_path}")
