"""Evaluate a single rule using LLM-as-a-judge against ground truth."""

from __future__ import annotations

import glob as _glob
import importlib
import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


def _load_predictions(rule_run_dir: str, question_slug: str, rule_name: str) -> dict[str, dict]:
    """Load prediction records indexed by doc_name, with slug fallback."""
    base_path = Path(rule_run_dir) / question_slug / f"{rule_name}_individual.json"
    if base_path.exists():
        records = json.loads(base_path.read_text(encoding="utf-8"))
        return {r["doc_name"]: r for r in records}

    # Fallback: strip trailing _<number> and retry with any matching folder
    stripped = re.sub(r"_\d+$", "", question_slug)
    if stripped != question_slug:
        stripped_path = Path(rule_run_dir) / stripped / f"{rule_name}_individual.json"
        if stripped_path.exists():
            records = json.loads(stripped_path.read_text(encoding="utf-8"))
            return {r["doc_name"]: r for r in records}

        pattern = str(Path(rule_run_dir) / f"{stripped}_*" / f"{rule_name}_individual.json")
        matches = sorted(_glob.glob(pattern))
        if matches:
            records = json.loads(Path(matches[-1]).read_text(encoding="utf-8"))
            return {r["doc_name"]: r for r in records}

    raise FileNotFoundError(
        f"Prediction file not found for '{rule_name}' in '{rule_run_dir}/{question_slug}/'. "
        "Run rule_apply_individual first."
    )


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


def eval_rule(
    rule_name: str,
    doc_names: list[str],
    question: str,
    question_slug: str,
    model_name: str = "gpt54",
    rule_run_dir: str = "results/financebench/lsf/single_cluster/llm/gpt54/one_shot/rule_run_individual",
    processing_dir: str = "data/financebench/processing",
    labels_file: str = "data/financebench/sample_labels.json",
    output_dir: str = "results/financebench/lsf/single_cluster/llm/gpt54/one_shot/eval_individual",
) -> dict:
    """Evaluate a rule against ground truth using LLM-as-a-judge.

    Returns aggregate metrics and per-document results.
    """
    eval_start = time.time()

    model_mod = importlib.import_module(f"models.{model_name}")

    # Step 1 — Load predictions
    predictions = _load_predictions(rule_run_dir, question_slug, rule_name)

    # Step 2 — Load ground truth labels
    labels: dict[str, dict] = json.loads(Path(labels_file).read_text(encoding="utf-8"))

    per_document: list[dict] = []
    judge_latencies: list[float] = []
    rule_apply_latencies: list[float] = []
    total_judge_input_tokens = 0
    total_judge_output_tokens = 0

    for doc_name in doc_names:
        pred_record = predictions.get(doc_name)

        if pred_record is not None:
            predicted_answer = pred_record.get("predicted_answer")
            retrieved_token_count = pred_record.get("retrieved_token_count", 0)
            rule_apply_latency = pred_record.get("latency_seconds", 0.0)
        else:
            predicted_answer = None
            retrieved_token_count = 0
            rule_apply_latency = 0.0

        rule_apply_latencies.append(rule_apply_latency)

        # Ground truth
        ground_truth = labels.get(doc_name + ".pdf", {}).get(question, None)

        # Total doc tokens
        doc_json_path = Path(processing_dir) / f"{doc_name}_reconstructed.json"
        if doc_json_path.exists():
            doc = json.loads(doc_json_path.read_text(encoding="utf-8"))
            total_doc_tokens = _count_tokens("\n".join(s.get("text", "") for s in doc.get("texts", [])))
        else:
            total_doc_tokens = None

        cost_ratio = (
            round(retrieved_token_count / total_doc_tokens, 6)
            if (total_doc_tokens and total_doc_tokens > 0)
            else None
        )

        # Step 3 — Judge
        if ground_truth is None:
            correct = False
            judge_input_tokens = 0
            judge_output_tokens = 0
            judge_latency = 0.0
        else:
            gt_str = json.dumps(ground_truth) if not isinstance(ground_truth, str) else ground_truth
            pred_str = str(predicted_answer) if predicted_answer is not None else "null"

            user_prompt = (
                f"Question: {question}\n"
                f"Ground Truth: {gt_str}\n"
                f"Predicted: {pred_str}"
            )

            t_judge = time.time()
            response = model_mod.client.chat.completions.create(
                model=model_mod.AZURE_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=10,
                temperature=0.0,
            )
            judge_latency = time.time() - t_judge

            judge_raw = (response.choices[0].message.content or "").strip()
            usage = response.usage
            judge_input_tokens = usage.prompt_tokens if usage else 0
            judge_output_tokens = usage.completion_tokens if usage else 0

            verdict = judge_raw.strip().lower()
            if verdict not in ("correct", "incorrect"):
                warnings.warn(f"Unexpected judge response for {doc_name}: '{judge_raw}'")
                correct = False
            else:
                correct = verdict == "correct"

        judge_latencies.append(judge_latency)
        total_judge_input_tokens += judge_input_tokens
        total_judge_output_tokens += judge_output_tokens

        per_document.append({
            "doc_name": doc_name,
            "predicted_answer": predicted_answer,
            "ground_truth": ground_truth,
            "correct": correct,
            "retrieved_token_count": retrieved_token_count,
            "total_doc_tokens": total_doc_tokens,
            "cost_ratio": cost_ratio,
            "judge_input_tokens": judge_input_tokens,
            "judge_output_tokens": judge_output_tokens,
            "judge_latency_seconds": round(judge_latency, 3),
            "rule_apply_latency_seconds": round(rule_apply_latency, 3),
        })

    n = len(doc_names)
    num_correct = sum(1 for r in per_document if r["correct"])
    accuracy = num_correct / n if n > 0 else 0.0

    valid_costs = [r["cost_ratio"] for r in per_document if r["cost_ratio"] is not None]
    avg_cost_ratio = sum(valid_costs) / len(valid_costs) if valid_costs else 0.0

    avg_retrieved = sum(r["retrieved_token_count"] for r in per_document) / n if n > 0 else 0.0

    avg_rule_apply_latency = sum(rule_apply_latencies) / n if n > 0 else 0.0

    judge_lats_nonzero = [l for l in judge_latencies if l > 0]
    avg_judge_latency = sum(judge_lats_nonzero) / len(judge_lats_nonzero) if judge_lats_nonzero else 0.0
    total_judge_latency = sum(judge_latencies)

    total_eval_latency = time.time() - eval_start

    result: dict[str, Any] = {
        "rule_name": rule_name,
        "question_slug": question_slug,
        "question": question,
        "strategy": "individual",
        "num_documents": n,
        "accuracy": round(accuracy, 4),
        "avg_cost_ratio": round(avg_cost_ratio, 6),
        "avg_retrieved_token_count": round(avg_retrieved, 2),
        "total_eval_latency_seconds": round(total_eval_latency, 3),
        "avg_rule_apply_latency_seconds": round(avg_rule_apply_latency, 3),
        "avg_judge_latency_seconds": round(avg_judge_latency, 3),
        "total_judge_latency_seconds": round(total_judge_latency, 3),
        "total_judge_input_tokens": total_judge_input_tokens,
        "total_judge_output_tokens": total_judge_output_tokens,
        "per_document": per_document,
    }

    out_path = Path(output_dir) / question_slug / f"{rule_name}_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result
