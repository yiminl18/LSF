"""Select a minimal-cost subset of rules whose merge accuracy matches a target."""

from __future__ import annotations

import importlib
import importlib.util
import json
import shutil
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
import sys
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ── Token counting ─────────────────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


# ── Rule loading ───────────────────────────────────────────────────────────────

def _load_rule_fn(rule_file: Path):
    spec = importlib.util.spec_from_file_location("_rule_mod", str(rule_file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        return next(v for k, v in vars(mod).items() if k.startswith("rule_") and callable(v))
    except StopIteration:
        raise ValueError(f"No rule_ function found in {rule_file}")


# ── Span union helper ──────────────────────────────────────────────────────────

def _union_spans(all_spans: list[dict], texts: list[dict]) -> list[dict]:
    text_positions: dict[int, int] = {id(s): i for i, s in enumerate(texts)}
    seen: set[int] = set()
    union: list[dict] = []
    for span in all_spans:
        idx = text_positions.get(id(span))
        if idx is None:
            try:
                idx = texts.index(span)
            except ValueError:
                idx = None
        if idx is None or idx not in seen:
            if idx is not None:
                seen.add(idx)
            union.append(span)
    return union


def _sort_spans(spans: list[dict], texts: list[dict]) -> list[dict]:
    text_positions: dict[int, int] = {id(s): i for i, s in enumerate(texts)}
    def key(span: dict) -> tuple:
        page = span.get("page_no", 0)
        structure = span.get("structure") or {}
        level_index = structure.get("level_index")
        if level_index is None:
            level_index = text_positions.get(id(span), 0)
        return (page, level_index)
    return sorted(spans, key=key)


# ── LLM prompts ────────────────────────────────────────────────────────────────

_QA_SYSTEM = (
    "You are a financial document QA assistant.\n"
    "You are given a passage extracted from a financial filing and a question.\n"
    "Answer the question using only the provided passage.\n"
    'If the passage does not contain enough information to answer, reply with "NOT FOUND".\n'
    "Return only the answer — a short value or phrase, not a full sentence."
)

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


# ── Core evaluation helper ─────────────────────────────────────────────────────

def evaluate_merge_accuracy(
    rule_names: list[str],
    documents: list[dict],
    ground_truth: dict,
    question: str,
    rule_folder: Path,
    model_mod,
) -> tuple[float, list[dict], int, int, int, int]:
    """Apply rule_names on every doc, call QA LLM + judge, return
    (accuracy, per_doc, qa_input_tokens, qa_output_tokens, judge_input_tokens, judge_output_tokens)."""
    num_correct = 0
    per_doc: list[dict] = []
    qa_input_tokens    = 0
    qa_output_tokens   = 0
    judge_input_tokens = 0
    judge_output_tokens = 0

    for doc in documents:
        texts = doc.get("texts", [])
        doc_name = doc.get("doc_name", doc.get("origin", {}).get("filename", "unknown"))
        if doc_name.endswith(".pdf"):
            doc_name = doc_name[:-4]

        # Apply each rule → collect spans
        all_spans: list[dict] = []
        for rule_name in rule_names:
            rule_file = rule_folder / f"{rule_name}.py"
            if not rule_file.exists():
                continue
            try:
                fn = _load_rule_fn(rule_file)
                spans = fn(doc)
                if spans:
                    all_spans.extend(spans)
            except Exception as e:
                warnings.warn(f"Rule {rule_name} error on {doc_name}: {e}")

        # Union, sort, concatenate
        union = _union_spans(all_spans, texts)
        sorted_s = _sort_spans(union, texts)
        retrieved_text = "\n\n".join(s["text"] for s in sorted_s) if sorted_s else ""
        retrieved_tokens = _count_tokens(retrieved_text)

        # QA LLM call
        t0 = time.time()
        qa_resp = model_mod.client.chat.completions.create(
            model=model_mod.AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _QA_SYSTEM},
                {"role": "user",   "content": f"Passage:\n{retrieved_text}\n\nQuestion: {question}"},
            ],
            max_completion_tokens=500,
            temperature=0.0,
        )
        predicted = (qa_resp.choices[0].message.content or "").strip() or None
        latency = round(time.time() - t0, 3)
        qa_input_tokens  += qa_resp.usage.prompt_tokens     if qa_resp.usage else 0
        qa_output_tokens += qa_resp.usage.completion_tokens if qa_resp.usage else 0

        # Judge LLM call
        gt_val = ground_truth.get(doc_name + ".pdf") or ground_truth.get(doc_name)
        correct = False
        if gt_val is not None and predicted is not None:
            gt_str   = json.dumps(gt_val) if not isinstance(gt_val, str) else gt_val
            pred_str = str(predicted)
            judge_resp = model_mod.client.chat.completions.create(
                model=model_mod.AZURE_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user",   "content": (
                        f"Question: {question}\n"
                        f"Ground Truth: {gt_str}\n"
                        f"Predicted: {pred_str}"
                    )},
                ],
                max_completion_tokens=10,
                temperature=0.0,
            )
            verdict = (judge_resp.choices[0].message.content or "").strip().lower()
            correct = verdict == "correct"
            judge_input_tokens  += judge_resp.usage.prompt_tokens     if judge_resp.usage else 0
            judge_output_tokens += judge_resp.usage.completion_tokens if judge_resp.usage else 0

        if correct:
            num_correct += 1

        per_doc.append({
            "doc_name":        doc_name,
            "predicted":       predicted,
            "ground_truth":    gt_val,
            "correct":         correct,
            "retrieved_tokens": retrieved_tokens,
            "latency_seconds": latency,
        })

    accuracy = num_correct / len(documents) if documents else 0.0
    return accuracy, per_doc, qa_input_tokens, qa_output_tokens, judge_input_tokens, judge_output_tokens


# ── Main function ──────────────────────────────────────────────────────────────

def rule_refine(
    rule_names: list[str],
    target_accuracy: float,
    question: str,
    question_slug: str,
    documents: list[dict],
    ground_truth: dict,
    rules_dir: str = "rules/llm/financebench",
    output_dir: str = "results/llm_rule_refine",
    model_name: str = "gpt54",
) -> dict:
    """Select a minimal-cost subset of rules whose merge accuracy matches target_accuracy."""

    import importlib as _imp
    model_mod = _imp.import_module(f"models.{model_name}")

    rule_folder = Path(rules_dir) / f"{question_slug}_llm"
    out_dir     = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    trace: list[dict] = []   # per-iteration log

    # ── Step 0: compute avg cost per rule (no LLM) ────────────────────────────
    avg_cost: dict[str, float] = {}
    valid_rules: list[str] = []

    for rule_name in rule_names:
        rule_file = rule_folder / f"{rule_name}.py"
        if not rule_file.exists():
            warnings.warn(f"Rule file missing, skipping: {rule_file}")
            continue
        try:
            fn = _load_rule_fn(rule_file)
        except Exception as e:
            warnings.warn(f"Cannot load {rule_name}: {e}")
            continue

        costs: list[float] = []
        for doc in documents:
            texts = doc.get("texts", [])
            total_text = "\n".join(s.get("text", "") for s in texts)
            total_tokens = _count_tokens(total_text)
            try:
                spans = fn(doc)
                retrieved_text = "\n\n".join(s["text"] for s in spans) if spans else ""
                retrieved_tokens = _count_tokens(retrieved_text)
            except Exception:
                retrieved_tokens = 0
            costs.append(retrieved_tokens / total_tokens if total_tokens > 0 else 0.0)

        avg_cost[rule_name] = mean(costs) if costs else 0.0
        valid_rules.append(rule_name)
        print(f"  [cost] {rule_name}: {avg_cost[rule_name]:.4f}", flush=True)

    if not valid_rules:
        raise ValueError("No valid rules found.")

    # ── Step 1: sort by avg cost ascending ────────────────────────────────────
    sorted_rules = sorted(valid_rules, key=lambda r: avg_cost[r])
    print(f"  [sorted] cheapest={sorted_rules[0]}  most_expensive={sorted_rules[-1]}", flush=True)

    # ── Compute avg cost of a rule set (token counting only) ──────────────────
    def _avg_cost_of_set(names: list[str]) -> float:
        if not names:
            return 0.0
        per_doc_costs: list[float] = []
        for doc in documents:
            texts = doc.get("texts", [])
            total_text = "\n".join(s.get("text", "") for s in texts)
            total_tokens = _count_tokens(total_text)
            all_spans: list[dict] = []
            for rn in names:
                rf = rule_folder / f"{rn}.py"
                if not rf.exists():
                    continue
                try:
                    fn = _load_rule_fn(rf)
                    spans = fn(doc)
                    if spans:
                        all_spans.extend(spans)
                except Exception:
                    pass
            union = _union_spans(all_spans, texts)
            retrieved_text = "\n\n".join(s["text"] for s in union) if union else ""
            retrieved_tokens = _count_tokens(retrieved_text)
            per_doc_costs.append(retrieved_tokens / total_tokens if total_tokens > 0 else 0.0)
        return mean(per_doc_costs) if per_doc_costs else 0.0

    avg_cost_all = _avg_cost_of_set(sorted_rules)
    print(f"  [baseline] avg_cost_all={avg_cost_all:.4f}  valid_rules={len(valid_rules)}", flush=True)

    # ── Step 2: exponential search ────────────────────────────────────────────
    total_llm_calls = 0
    exp_steps = 0
    candidate = sorted_rules  # fallback
    total_qa_input_tokens    = 0
    total_qa_output_tokens   = 0
    total_judge_input_tokens = 0
    total_judge_output_tokens = 0

    k = 1
    while k <= len(sorted_rules):
        subset = sorted_rules[:k]
        t0 = time.time()
        acc, _, qa_in, qa_out, judge_in, judge_out = evaluate_merge_accuracy(subset, documents, ground_truth, question, rule_folder, model_mod)
        iter_latency = round(time.time() - t0, 3)
        total_llm_calls += 2 * len(documents)
        total_qa_input_tokens    += qa_in
        total_qa_output_tokens   += qa_out
        total_judge_input_tokens  += judge_in
        total_judge_output_tokens += judge_out
        exp_steps += 1

        cost = _avg_cost_of_set(subset)
        action = "continue"
        if acc >= target_accuracy:
            candidate = subset
            action = "candidate_found"
        elif k == len(sorted_rules):
            candidate = sorted_rules
            action = "all_rules"
            warnings.warn(f"Target accuracy {target_accuracy:.2f} not achievable with any subset; using all rules.")

        trace.append({
            "phase":        "exp_search",
            "step":         exp_steps,
            "k":            k,
            "rules":        subset,
            "rules_count":  len(subset),
            "accuracy":     round(acc, 4),
            "avg_cost_ratio": round(cost, 6),
            "latency_seconds": iter_latency,
            "input_tokens":  qa_in + judge_in,
            "output_tokens": qa_out + judge_out,
            "action":       action,
        })
        print(f"    [exp k={k}] acc={acc:.2f} target={target_accuracy:.2f} cost={cost:.4f} latency={iter_latency:.1f}s action={action}", flush=True)

        if action in ("candidate_found", "all_rules"):
            break

        k = min(k * 2, len(sorted_rules))

    # ── Step 3: backward linear pruning ───────────────────────────────────────
    pruning_steps = 0
    refined = list(candidate)

    i = len(refined) - 1
    while i > 0:
        subset = refined[:i] + refined[i+1:]
        t0 = time.time()
        acc, _, qa_in, qa_out, judge_in, judge_out = evaluate_merge_accuracy(subset, documents, ground_truth, question, rule_folder, model_mod)
        iter_latency = round(time.time() - t0, 3)
        total_llm_calls += 2 * len(documents)
        total_qa_input_tokens    += qa_in
        total_qa_output_tokens   += qa_out
        total_judge_input_tokens  += judge_in
        total_judge_output_tokens += judge_out
        pruning_steps += 1

        cost = _avg_cost_of_set(subset)
        rule_removed = refined[i]
        if acc >= target_accuracy:
            refined.pop(i)
            action = "removed"
        else:
            action = "kept"

        trace.append({
            "phase":          "pruning",
            "step":           pruning_steps,
            "i":              i,
            "rule_tested":    rule_removed,
            "rules":          list(subset),
            "rules_count":    len(subset),
            "accuracy":       round(acc, 4),
            "avg_cost_ratio": round(cost, 6),
            "latency_seconds": iter_latency,
            "input_tokens":   qa_in + judge_in,
            "output_tokens":  qa_out + judge_out,
            "action":         action,
        })
        print(f"    [prune i={i}] remove={rule_removed} acc={acc:.2f} cost={cost:.4f} latency={iter_latency:.1f}s action={action}", flush=True)

        i -= 1

    # ── Final evaluation on selected rules ────────────────────────────────────
    t0 = time.time()
    final_acc, final_per_doc, qa_in, qa_out, judge_in, judge_out = evaluate_merge_accuracy(
        refined, documents, ground_truth, question, rule_folder, model_mod
    )
    final_latency = round(time.time() - t0, 3)
    total_llm_calls += 2 * len(documents)
    total_qa_input_tokens    += qa_in
    total_qa_output_tokens   += qa_out
    total_judge_input_tokens  += judge_in
    total_judge_output_tokens += judge_out

    avg_cost_selected = _avg_cost_of_set(refined)
    trace.append({
        "phase":          "final",
        "step":           1,
        "rules":          refined,
        "rules_count":    len(refined),
        "accuracy":       round(final_acc, 4),
        "avg_cost_ratio": round(avg_cost_selected, 6),
        "latency_seconds": final_latency,
        "input_tokens":   qa_in + judge_in,
        "output_tokens":  qa_out + judge_out,
        "action":         "final_eval",
    })

    total_latency = round(time.time() - t_start, 3)
    avg_latency   = round(total_latency / len(documents), 3) if documents else 0.0
    cost_reduction = round(1.0 - avg_cost_selected / avg_cost_all, 6) if avg_cost_all > 0 else 0.0

    # ── Enrich per_doc with token counts ──────────────────────────────────────
    for entry in final_per_doc:
        doc_name = entry["doc_name"]
        doc = next((d for d in documents
                    if d.get("doc_name", d.get("origin", {}).get("filename", "")).replace(".pdf", "") == doc_name), None)
        if doc:
            total_text   = "\n".join(s.get("text", "") for s in doc.get("texts", []))
            total_tokens = _count_tokens(total_text)
            entry["total_doc_tokens"] = total_tokens
            entry["cost_ratio"] = round(entry["retrieved_tokens"] / total_tokens, 6) if total_tokens > 0 else 0.0

    # ── Copy selected rule files ───────────────────────────────────────────────
    sel_dir = out_dir / question_slug
    sel_dir.mkdir(parents=True, exist_ok=True)
    for rn in refined:
        src = rule_folder / f"{rn}.py"
        if src.exists():
            shutil.copy2(src, sel_dir / f"{rn}.py")

    # ── Write result JSON ──────────────────────────────────────────────────────
    result: dict[str, Any] = {
        "question":              question,
        "question_slug":         question_slug,
        "timestamp":             timestamp,
        "model":                 model_name,
        "num_documents":         len(documents),
        "target_accuracy":       target_accuracy,
        "all_rules_count":       len(valid_rules),
        "selected_rules_count":  len(refined),
        "selected_rules":        refined,
        "merge_accuracy":        round(final_acc, 4),
        "avg_cost_ratio":        round(avg_cost_selected, 6),
        "avg_cost_ratio_all_rules": round(avg_cost_all, 6),
        "cost_reduction_ratio":  cost_reduction,
        "avg_latency_seconds":   avg_latency,
        "total_latency_seconds": total_latency,
        "total_llm_calls":       total_llm_calls,
        "exponential_search_steps": exp_steps,
        "pruning_steps":         pruning_steps,
        "qa_input_tokens":       total_qa_input_tokens,
        "qa_output_tokens":      total_qa_output_tokens,
        "judge_input_tokens":    total_judge_input_tokens,
        "judge_output_tokens":   total_judge_output_tokens,
        "total_input_tokens":    total_qa_input_tokens + total_judge_input_tokens,
        "total_output_tokens":   total_qa_output_tokens + total_judge_output_tokens,
        "per_doc":               final_per_doc,
    }

    result_path = out_dir / f"{question_slug}_refine.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Write trace JSON ───────────────────────────────────────────────────────
    trace_path = out_dir / f"{question_slug}_trace.json"
    trace_out = {
        "question":           question,
        "question_slug":      question_slug,
        "timestamp":          timestamp,
        "target_accuracy":    target_accuracy,
        "all_rules_count":    len(valid_rules),
        "sorted_rules":       sorted_rules,
        "rule_costs":         {r: round(avg_cost[r], 6) for r in sorted_rules},
        "total_latency_seconds": total_latency,
        "total_input_tokens":  total_qa_input_tokens + total_judge_input_tokens,
        "total_output_tokens": total_qa_output_tokens + total_judge_output_tokens,
        "iterations":         trace,
    }
    trace_path.write_text(json.dumps(trace_out, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Update summary.json ───────────────────────────────────────────────────
    summary_path = out_dir / "summary.json"
    summary: list[dict] = []
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    summary_entry = {
        "question":              question,
        "question_slug":         question_slug,
        "target_accuracy":       target_accuracy,
        "merge_accuracy":        round(final_acc, 4),
        "all_rules_count":       len(valid_rules),
        "selected_rules_count":  len(refined),
        "avg_cost_ratio_selected": round(avg_cost_selected, 6),
        "avg_cost_ratio_all":    round(avg_cost_all, 6),
        "cost_reduction_ratio":  cost_reduction,
        "total_llm_calls":       total_llm_calls,
    }

    updated = False
    for i, e in enumerate(summary):
        if e.get("question_slug") == question_slug:
            summary[i] = summary_entry
            updated = True
            break
    if not updated:
        summary.append(summary_entry)

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return result
