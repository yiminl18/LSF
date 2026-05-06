"""Evaluate merged refined rules on 50 unsampled docs for every sample query."""

from __future__ import annotations

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

from rule_apply_merge import rule_apply_merge
import importlib
model_mod = importlib.import_module("models.gpt54")

QUERIES_FILE   = "data/financebench/sample_queries.txt"
LABELS_FILE    = "data/financebench/unsampled_doc_labels.json"
PROCESSING_DIR = "data/financebench/processing"
RULES_BASE_DIR = "rules/financebench_single_cluster/llm/gpt54/refine"
MERGE_RUN_DIR  = "results/financebench_single_cluster/llm/gpt54/refine/rule_run_merge_unsampled"
OUTPUT_DIR     = "results/financebench_single_cluster/llm/gpt54/refine/eval_merge"

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


def make_slug(q: str) -> str:
    s = q.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60]


def count_tokens(text: str) -> int:
    return len(text.split())


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


# ── Load questions and labels ──────────────────────────────────────────────────
questions = [l.strip() for l in open(QUERIES_FILE) if l.strip()]
labels: dict[str, dict] = json.loads(Path(LABELS_FILE).read_text(encoding="utf-8"))
doc_names = [k.replace(".pdf", "") for k in labels.keys()]
print(f"Questions: {len(questions)}  |  Docs: {len(doc_names)}\n")

# ── Load all 50 documents upfront ─────────────────────────────────────────────
doc_map: dict[str, dict] = {}
for pdf_key in labels:
    doc_name = pdf_key.replace(".pdf", "")
    path = Path(PROCESSING_DIR) / f"{doc_name}_reconstructed.json"
    if path.exists():
        doc_map[doc_name] = json.loads(path.read_text(encoding="utf-8"))
    else:
        print(f"WARNING: missing {path}")
print(f"Documents loaded: {len(doc_map)}\n")

# ── Precompute total token counts ──────────────────────────────────────────────
doc_total_tokens: dict[str, int] = {
    doc_name: count_tokens("\n".join(s.get("text", "") for s in doc.get("texts", [])))
    for doc_name, doc in doc_map.items()
}

# ── Ensure output dirs ─────────────────────────────────────────────────────────
Path(MERGE_RUN_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ── Main loop ──────────────────────────────────────────────────────────────────
for question in questions:
    slug          = make_slug(question)
    question_slug = f"{slug}_10"
    rule_slug     = f"{slug}_10_refined"
    rule_folder   = Path(RULES_BASE_DIR) / rule_slug
    out_file      = Path(OUTPUT_DIR) / f"{question_slug}_unsampled_refined.json"

    if out_file.exists():
        print(f"SKIP (exists): {out_file}")
        continue

    if not rule_folder.is_dir():
        print(f"SKIP (no refined rules): {rule_folder}")
        continue

    rule_names = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(rule_folder)
        if f.startswith("rule_") and f.endswith(".py")
    )
    if not rule_names:
        print(f"SKIP (empty): {rule_folder}")
        continue

    print(f"\nQuestion: {question}  rules={len(rule_names)}  docs={len(doc_map)}")

    # derive run file path (rule_apply_merge writes here)
    rule_set_slug = "__".join(sorted(rule_names))[:120]
    run_file = Path(MERGE_RUN_DIR) / rule_slug / f"{rule_set_slug}_merge.json"

    # load already-completed doc_names
    already_done: set[str] = set()
    cached: dict[str, dict] = {}
    if run_file.exists():
        try:
            existing = json.loads(run_file.read_text(encoding="utf-8"))
            for r in existing:
                dn = r.get("doc_name", "")
                already_done.add(dn)
                cached[dn] = r
        except Exception:
            pass
    if already_done:
        print(f"  (resuming — {len(already_done)} docs already cached)")

    try:
        run_results: list[dict] = []

        for doc_name, document in doc_map.items():
            if doc_name in already_done:
                r = cached[doc_name]
                run_results.append({
                    "doc_name":         doc_name,
                    "predicted":        r.get("predicted_answer"),
                    "latency_seconds":  r.get("latency_seconds", 0.0),
                    "retrieved_tokens": r.get("retrieved_token_count", 0),
                    "input_tokens":     r.get("input_tokens", 0),
                })
                continue

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

        # ── Judge each prediction ──────────────────────────────────────────────
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

        # ── Aggregates ─────────────────────────────────────────────────────────
        n              = len(per_doc_out)
        n_correct      = sum(r["correct"] for r in per_doc_out)
        accuracy       = round(n_correct / n, 4) if n else 0.0
        avg_latency    = round(mean(r["latency_seconds"]  for r in per_doc_out), 3)
        avg_retrieved  = round(mean(r["retrieved_tokens"] for r in per_doc_out), 1)
        avg_input_tok  = round(mean(r["input_tokens"]     for r in per_doc_out), 1)
        avg_cost_ratio = round(mean(cost_ratios), 6)

        print(f"  accuracy={accuracy:.2f} ({n_correct}/{n})  avg_retrieved={avg_retrieved:.1f}  cost={avg_cost_ratio:.4f}")

        per_q = {
            "question":       question,
            "question_slug":  question_slug,
            "rule_slug":      rule_slug,
            "split":          "unsampled_refined",
            "n":              n,
            "n_correct":      n_correct,
            "accuracy":       accuracy,
            "avg_latency":    avg_latency,
            "avg_retrieved":  avg_retrieved,
            "avg_input_tok":  avg_input_tok,
            "avg_cost_ratio": avg_cost_ratio,
            "per_doc":        per_doc_out,
        }
        out_file.write_text(json.dumps(per_q, indent=2, ensure_ascii=False), encoding="utf-8")

        # ── Update summary.json ────────────────────────────────────────────────
        summary_path = Path(OUTPUT_DIR) / "summary.json"
        summary: list[dict] = []
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        entry = {
            "question":       question,
            "question_slug":  question_slug,
            "rule_slug":      rule_slug,
            "split":          "unsampled_refined",
            "n":              n,
            "n_correct":      n_correct,
            "accuracy":       accuracy,
            "avg_latency":    avg_latency,
            "avg_retrieved":  avg_retrieved,
            "avg_input_tok":  avg_input_tok,
            "avg_cost_ratio": avg_cost_ratio,
        }

        matched = False
        for e in summary:
            if e.get("question_slug") == question_slug:
                e.update(entry)
                matched = True
                break
        if not matched:
            summary.append(entry)

        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    except Exception as e:
        import traceback
        print(f"  FATAL ERROR for '{question}': {e}")
        traceback.print_exc()
        continue

# ── Final summary table ────────────────────────────────────────────────────────
print(f"\n{'Question':<55} {'Acc':>6}  {'AvgRetr':>8}  {'CostRatio':>10}")
print("-" * 85)
summary_path = Path(OUTPUT_DIR) / "summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for s in summary:
        print(
            f"  {s['question'][:53]:<53}  {s.get('accuracy',0):>5.2f}  "
            f"{s.get('avg_retrieved',0):>8.1f}  {s.get('avg_cost_ratio',0):>10.5f}"
        )

print(f"\nResults: {OUTPUT_DIR}/")
