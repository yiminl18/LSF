"""Result aggregation module: build per-query and summary JSON output."""


def build_query_result(
    q_idx: int,
    question: str,
    doc_results: list[dict],
    eval_detail: dict,
    train_time_ms: float,
    eval_time_ms: float,
) -> dict:
    """
    Build the complete result JSON for a single question.

    Args:
        q_idx: question index
        question: question text
        doc_results: per-doc result list, each entry contains document/gen_time_ms/
                     judge_time_ms/gen_cost_usd/judge_cost_usd/accuracy/
                     generated_answer/judge_result/gen_cache_hit/judge_cache_hit/top_k_used
        eval_detail: per-question detail from evaluate_models output
        train_time_ms: per-question share of total training time (ms)
        eval_time_ms: per-question share of total eval time (ms)

    Returns:
        Complete per-query result dict.
    """
    documents = []
    for doc in doc_results:
        gen_time = doc["gen_time_ms"]
        judge_time = doc["judge_time_ms"]
        gen_cost = doc["gen_cost_usd"]
        judge_cost = doc["judge_cost_usd"]

        total_time = train_time_ms + eval_time_ms + gen_time + judge_time
        total_cost = gen_cost + judge_cost  # train/eval cost = 0

        documents.append({
            "document": doc["document"],
            "total_time_ms": round(total_time, 2),
            "total_cost_usd": round(total_cost, 6),
            "training_time_ms": round(train_time_ms, 2),
            "eval_time_ms": round(eval_time_ms, 2),
            "gen_time_ms": round(gen_time, 2),
            "judge_time_ms": round(judge_time, 2),
            "train_cost_usd": 0.0,
            "eval_cost_usd": 0.0,
            "gen_cost_usd": round(gen_cost, 6),
            "judge_cost_usd": round(judge_cost, 6),
            "accuracy": doc["accuracy"],
            "generated_answer": doc["generated_answer"],
            "judge_result": doc["judge_result"],
            "gen_cache_hit": doc["gen_cache_hit"],
            "judge_cache_hit": doc["judge_cache_hit"],
            "top_k_used": doc["top_k_used"],
        })

    # Aggregate stats (accuracy<0 excluded: -1=no GT, -2=API error).
    judged = [d for d in doc_results if d["accuracy"] >= 0]
    skipped = len(doc_results) - len(judged)
    correct = sum(1 for d in judged if d["accuracy"] == 1)
    accuracy = correct / len(judged) if judged else 0.0

    gen_times = [d["gen_time_ms"] for d in doc_results]
    judge_times = [d["judge_time_ms"] for d in doc_results]

    aggregated = {
        "accuracy": round(accuracy, 4),
        "num_judged": len(judged),
        "num_skipped_no_gt": skipped,
        "mean_gen_time_ms": round(sum(gen_times) / len(gen_times), 2) if gen_times else 0.0,
        "mean_judge_time_ms": round(sum(judge_times) / len(judge_times), 2) if judge_times else 0.0,
        "total_gen_cost_usd": round(sum(d["gen_cost_usd"] for d in doc_results), 6),
        "total_judge_cost_usd": round(sum(d["judge_cost_usd"] for d in doc_results), 6),
    }

    # Eval ranking details.
    eval_ranking_details: dict = {}
    if eval_detail:
        if "target_docs_details" in eval_detail:
            eval_ranking_details["target_docs_details"] = eval_detail["target_docs_details"]
        if "metrics" in eval_detail:
            eval_ranking_details["metrics"] = eval_detail["metrics"]

    return {
        "q_idx": q_idx,
        "question": question,
        "num_docs": len(doc_results),
        "documents": documents,
        "eval_ranking_details": eval_ranking_details,
        "aggregated": aggregated,
    }


def build_summary(query_results: list[dict], meta: dict) -> dict:
    """
    Build the complete output for summary.json.

    Args:
        query_results: list of dicts returned by build_query_result
        meta: metadata dict; in addition to dataset/experiment etc., may include
              training_time_ms / eval_time_ms / num_questions_skipped
              (these internal fields are consumed and removed from the output meta)

    Returns:
        summary dict
    """
    timing_notes = {
        "training_time": "actual per-question wall-clock from train worker",
        "eval_time": "actual per-question wall-clock from eval worker",
        "gen_time": "actual LLM API latency per call (from cache if cache_hit)",
        "judge_time": "actual LLM API latency per call (from cache if cache_hit)",
    }

    # Extract timing fields from meta (not included in summary.meta output).
    train_time = meta.get("training_time_ms", 0.0)
    eval_time = meta.get("eval_time_ms", 0.0)
    num_questions_skipped = meta.get("num_questions_skipped", 0)

    _INTERNAL_KEYS = {"training_time_ms", "eval_time_ms", "num_questions_skipped"}
    output_meta = {k: v for k, v in meta.items() if k not in _INTERNAL_KEYS}

    # Exact per-doc aggregation.
    all_judged = 0
    all_correct = 0
    num_docs_total = 0
    total_gen_time = 0.0
    total_judge_time = 0.0
    total_gen_cost = 0.0
    total_judge_cost = 0.0

    per_question: list[dict] = []

    for qr in query_results:
        num_docs_total += qr["num_docs"]

        for doc in qr["documents"]:
            total_gen_time += doc["gen_time_ms"]
            total_judge_time += doc["judge_time_ms"]
            total_gen_cost += doc["gen_cost_usd"]
            total_judge_cost += doc["judge_cost_usd"]

            if doc["accuracy"] >= 0:
                all_judged += 1
                if doc["accuracy"] == 1:
                    all_correct += 1

        per_question.append({
            "q_idx": qr["q_idx"],
            "accuracy": qr["aggregated"]["accuracy"],
            "num_docs": qr["num_docs"],
            "num_judged": qr["aggregated"]["num_judged"],
            "gen_cost_usd": round(qr["aggregated"]["total_gen_cost_usd"], 6),
            "judge_cost_usd": round(qr["aggregated"]["total_judge_cost_usd"], 6),
        })

    total_cost = total_gen_cost + total_judge_cost
    total_time = train_time + eval_time + total_gen_time + total_judge_time
    accuracy = all_correct / all_judged if all_judged else 0.0

    overall = {
        "accuracy": round(accuracy, 4),
        "num_questions": len(query_results),
        "num_questions_skipped": num_questions_skipped,
        "num_docs_total": num_docs_total,
        "total_time_ms": round(total_time, 2),
        "total_cost_usd": round(total_cost, 6),
        "training_time_ms": round(train_time, 2),
        "eval_time_ms": round(eval_time, 2),
        "gen_time_ms": round(total_gen_time, 2),
        "judge_time_ms": round(total_judge_time, 2),
        "train_cost_usd": 0.0,
        "eval_cost_usd": 0.0,
        "gen_cost_usd": round(total_gen_cost, 6),
        "judge_cost_usd": round(total_judge_cost, 6),
    }

    return {
        "meta": output_meta,
        "timing_notes": timing_notes,
        "overall": overall,
        "per_question": per_question,
    }
