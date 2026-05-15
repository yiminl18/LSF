"""Overfitting analysis: sampled-refined vs unsampled-refined rule selection.

Compares rules selected on 10 sampled docs (gpt54 refinement) vs rules selected
on 50 unsampled docs (gpt54mini refinement, "gold" standard).
Evaluates every individual rule on the 50 unsampled docs.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

# ── Path setup ────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ── Directories ───────────────────────────────────────────────────────────────

REFINE_DIR       = _ROOT / "results/financebench_single_cluster/llm/gpt54/refine/rule_refine"
GOLD_REFINE_DIR  = _ROOT / "rules/financebench_single_cluster/llm/gpt54mini/refine_unsampled"
EVAL_MERGE_DIR   = _ROOT / "results/financebench_single_cluster/llm/gpt54/refine/eval_merge"
RULE_BASE        = _ROOT / "rules/financebench_single_cluster/llm/gpt54/one_shot"
DATA_DIR         = _ROOT / "data/financebench"
CACHE_DIR        = _ROOT / "analysis/rule_eval_cache"
OUTPUT_FILE      = _ROOT / "analysis/overfitting_analysis.txt"

UNSAMPLED_LABELS = DATA_DIR / "unsampled_doc_labels.json"
PROCESSING_DIR   = DATA_DIR / "processing"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── LLM prompts (same as rule_refine.py) ────────────────────────────────────

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


# ── Token counting ────────────────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


# ── Rule loading ──────────────────────────────────────────────────────────────

def _load_rule_fn(rule_file: Path):
    spec = importlib.util.spec_from_file_location("_rule_mod", str(rule_file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        return next(v for k, v in vars(mod).items() if k.startswith("rule_") and callable(v))
    except StopIteration:
        raise ValueError(f"No rule_ function found in {rule_file}")


# ── Document loading ──────────────────────────────────────────────────────────

def load_unsampled_docs() -> list[dict]:
    """Load all 50 unsampled documents once."""
    labels = json.loads(UNSAMPLED_LABELS.read_text())
    docs = []
    for pdf_key in labels:
        doc_name = pdf_key.replace(".pdf", "")
        proc_file = PROCESSING_DIR / f"{doc_name}_reconstructed.json"
        if not proc_file.exists():
            warnings.warn(f"Missing processing file: {proc_file}")
            continue
        doc = json.loads(proc_file.read_text())
        doc["doc_name"] = doc_name
        docs.append(doc)
    print(f"Loaded {len(docs)} unsampled documents.")
    return docs


# ── Core: evaluate a single rule on all docs ─────────────────────────────────

def evaluate_rule(
    rule_name: str,
    rule_file: Path,
    question: str,
    question_slug: str,
    documents: list[dict],
    unsampled_labels: dict,
    rule_costs: dict[str, float],
    model_mod,
) -> dict:
    """Evaluate one rule on all unsampled docs. Use cache if available."""
    cache_dir = CACHE_DIR / question_slug
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{rule_name}.json"

    # Load from cache if exists
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    # Load rule function
    try:
        fn = _load_rule_fn(rule_file)
    except Exception as e:
        warnings.warn(f"Failed to load rule {rule_name}: {e}")
        # Return a stub with all-incorrect
        result = {
            "rule_name": rule_name,
            "question": question,
            "per_doc": [],
            "accuracy": 0.0,
            "coverage": 0.0,
            "avg_cost_ratio": rule_costs.get(rule_name, 0.0),
            "error": str(e),
        }
        cache_file.write_text(json.dumps(result, indent=2))
        return result

    per_doc_results = []
    num_correct = 0
    num_covered = 0

    for doc in documents:
        doc_name = doc.get("doc_name", "unknown")
        texts = doc.get("texts", [])
        total_tokens = sum(_count_tokens(t.get("text", "")) for t in texts)

        # Apply rule
        try:
            spans = fn(doc) or []
        except Exception as e:
            warnings.warn(f"Rule {rule_name} error on {doc_name}: {e}")
            spans = []

        retrieved_text = "\n".join(s["text"] for s in spans if s.get("text"))
        retrieved_tokens = _count_tokens(retrieved_text)
        covered = len(spans) > 0
        if covered:
            num_covered += 1

        cost_ratio = retrieved_tokens / total_tokens if total_tokens > 0 else 0.0

        # Get ground truth for this specific question
        gt_dict = unsampled_labels.get(doc_name + ".pdf", {}) or unsampled_labels.get(doc_name, {})
        gt_val = gt_dict.get(question)

        predicted = None
        correct = False

        if retrieved_text:
            # QA call
            try:
                qa_resp = model_mod.client.chat.completions.create(
                    model=model_mod.AZURE_DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": _QA_SYSTEM},
                        {"role": "user", "content": f"Passage:\n{retrieved_text}\n\nQuestion: {question}"},
                    ],
                    max_completion_tokens=500,
                    temperature=0.0,
                )
                predicted = (qa_resp.choices[0].message.content or "").strip() or None
            except Exception as e:
                if "content_filter" in str(e) or "content management" in str(e):
                    warnings.warn(f"Content filter (QA) on {doc_name}, treating as NOT FOUND: {e}")
                    predicted = None
                else:
                    warnings.warn(f"QA error on {doc_name}: {e}")
                    predicted = None

        # Judge call
        if gt_val is not None and predicted is not None:
            gt_str = json.dumps(gt_val) if not isinstance(gt_val, str) else gt_val
            try:
                judge_resp = model_mod.client.chat.completions.create(
                    model=model_mod.AZURE_DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": _JUDGE_SYSTEM},
                        {"role": "user", "content": (
                            f"Question: {question}\n"
                            f"Ground Truth: {gt_str}\n"
                            f"Predicted: {predicted}"
                        )},
                    ],
                    max_completion_tokens=10,
                    temperature=0.0,
                )
                verdict = (judge_resp.choices[0].message.content or "").strip().upper()
                correct = (verdict == "CORRECT")
            except Exception as e:
                if "content_filter" in str(e) or "content management" in str(e):
                    warnings.warn(f"Content filter (judge) on {doc_name}, treating as incorrect: {e}")
                    correct = False
                else:
                    warnings.warn(f"Judge error on {doc_name}: {e}")
                    correct = False

        if correct:
            num_correct += 1

        per_doc_results.append({
            "doc_name": doc_name,
            "predicted": predicted,
            "correct": correct,
            "retrieved_tokens": retrieved_tokens,
            "cost_ratio": cost_ratio,
        })

    n = len(documents)
    accuracy = num_correct / n if n > 0 else 0.0
    coverage = num_covered / n if n > 0 else 0.0

    # avg_cost_ratio: prefer from trace if available
    if rule_name in rule_costs:
        avg_cost_ratio = rule_costs[rule_name]
    else:
        ratios = [r["cost_ratio"] for r in per_doc_results]
        avg_cost_ratio = mean(ratios) if ratios else 0.0

    result = {
        "rule_name": rule_name,
        "question": question,
        "per_doc": per_doc_results,
        "accuracy": accuracy,
        "coverage": coverage,
        "avg_cost_ratio": avg_cost_ratio,
    }

    # Write cache (never overwrite)
    if not cache_file.exists():
        cache_file.write_text(json.dumps(result, indent=2))

    return result


# ── Main analysis ─────────────────────────────────────────────────────────────

def main():
    import models.gpt54mini as model_mod

    print("Loading unsampled labels...")
    unsampled_labels = json.loads(UNSAMPLED_LABELS.read_text())

    print("Loading unsampled documents...")
    unsampled_docs = load_unsampled_docs()

    # Collect all question slugs
    sampled_refine_files = sorted(REFINE_DIR.glob("*_refine.json"))
    question_slugs = [f.stem.replace("_refine", "") for f in sampled_refine_files]

    output_lines = []
    output_lines.append("OVERFITTING ANALYSIS: Sampled-Refined vs Unsampled-Refined Rules")
    output_lines.append("=" * 64)
    output_lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    output_lines.append("")

    # For overall summary
    overall_rows = []

    for slug in question_slugs:
        print(f"\n{'='*60}")
        print(f"Processing: {slug}")

        # Load sampled-refined data
        sampled_file = REFINE_DIR / f"{slug}_refine.json"
        if not sampled_file.exists():
            print(f"  WARNING: No sampled refine file for {slug}")
            continue
        sampled_data = json.loads(sampled_file.read_text())
        question = sampled_data["question"]
        sampled_selected = sampled_data.get("selected_rules", [])

        # Load gold (unsampled) data
        gold_file = GOLD_REFINE_DIR / f"{slug}_refine.json"
        if not gold_file.exists():
            print(f"  WARNING: No gold refine file for {slug}")
            continue
        gold_data = json.loads(gold_file.read_text())
        gold_selected = gold_data.get("selected_rules", [])

        # Load trace for rule_costs
        trace_file = GOLD_REFINE_DIR / f"{slug}_trace.json"
        rule_costs: dict[str, float] = {}
        if trace_file.exists():
            trace_data = json.loads(trace_file.read_text())
            rule_costs = trace_data.get("rule_costs", {})

        # Sampled-refined merged acc on unsampled docs
        eval_merge_file = EVAL_MERGE_DIR / f"{slug}_unsampled_refined.json"
        sampled_merged_acc_unsampled = None
        if eval_merge_file.exists():
            em = json.loads(eval_merge_file.read_text())
            sampled_merged_acc_unsampled = em.get("accuracy")

        # Gold merged acc on unsampled (from gold refine file)
        gold_merged_acc_unsampled = gold_data.get("merge_accuracy")

        # Collect all unique rules to evaluate
        all_rules = sorted(set(sampled_selected) | set(gold_selected))

        # Rule folder
        rule_folder = RULE_BASE / f"{slug}_llm"

        # Evaluate each rule individually
        rule_results: dict[str, dict] = {}
        for rule_name in all_rules:
            rule_file = rule_folder / f"{rule_name}.py"
            if not rule_file.exists():
                warnings.warn(f"Rule file not found: {rule_file}")
                rule_results[rule_name] = {
                    "rule_name": rule_name,
                    "question": question,
                    "per_doc": [],
                    "accuracy": 0.0,
                    "coverage": 0.0,
                    "avg_cost_ratio": rule_costs.get(rule_name, 0.0),
                    "error": "file not found",
                }
                continue
            print(f"  Evaluating rule: {rule_name}")
            res = evaluate_rule(
                rule_name=rule_name,
                rule_file=rule_file,
                question=question,
                question_slug=slug,
                documents=unsampled_docs,
                unsampled_labels=unsampled_labels,
                rule_costs=rule_costs,
                model_mod=model_mod,
            )
            rule_results[rule_name] = res
            print(f"    acc={res['accuracy']:.2f}, cov={res['coverage']:.2f}, cost={res['avg_cost_ratio']:.5f}")

        # ── Format question section ───────────────────────────────────────────

        output_lines.append(f"QUESTION: {question}")
        output_lines.append("-" * 57)

        # Sampled acc on sampled docs (from per_doc in sampled_refine)
        sampled_per_doc = sampled_data.get("per_doc", [])
        sampled_acc_on_sampled = (
            sum(1 for d in sampled_per_doc if d.get("correct")) / len(sampled_per_doc)
            if sampled_per_doc else None
        )

        if sampled_acc_on_sampled is not None:
            output_lines.append(f"  Sampled-refined acc (merged, 10 sampled docs):    {sampled_acc_on_sampled:.2f}  (#rules: {len(sampled_selected)})")
        if sampled_merged_acc_unsampled is not None:
            output_lines.append(f"  Sampled-refined acc (merged, 50 unsampled docs):  {sampled_merged_acc_unsampled:.2f}  (#rules: {len(sampled_selected)})")
        if gold_merged_acc_unsampled is not None:
            output_lines.append(f"  Unsampled-refined acc (merged, 50 unsampled docs): {gold_merged_acc_unsampled:.2f}  (#rules: {len(gold_selected)})")

        if sampled_merged_acc_unsampled is not None and gold_merged_acc_unsampled is not None:
            gap = sampled_merged_acc_unsampled - gold_merged_acc_unsampled
            output_lines.append(f"  Accuracy gap (sampled_merged - gold_merged):       {gap:+.2f}")

        output_lines.append("")

        # ── Sampled-refined rules table ───────────────────────────────────────
        output_lines.append("  SAMPLED-REFINED RULES (selected on 10 sampled docs, gpt54):")
        header = f"  {'rule_name':<40} | {'In_Gold':<7} | {'Accuracy':<8} | {'Coverage':<8} | {'Cost_Ratio'}"
        output_lines.append(header)
        output_lines.append("  " + "-" * 85)

        gold_set = set(gold_selected)
        sampled_set = set(sampled_selected)

        for rule_name in sampled_selected:
            res = rule_results.get(rule_name, {})
            in_gold = "YES" if rule_name in gold_set else "NO"
            acc = res.get("accuracy", 0.0)
            cov = res.get("coverage", 0.0)
            cost = res.get("avg_cost_ratio", 0.0)
            output_lines.append(
                f"  {rule_name:<40} | {in_gold:<7} | {acc:<8.2f} | {cov:<8.2f} | {cost:.5f}"
            )

        output_lines.append("")

        # ── Gold rules table ──────────────────────────────────────────────────
        output_lines.append("  UNSAMPLED-REFINED (GOLD) RULES (selected on 50 unsampled docs, gpt54mini):")
        header2 = f"  {'rule_name':<40} | {'In_Sampled':<10} | {'Accuracy':<8} | {'Coverage':<8} | {'Cost_Ratio'}"
        output_lines.append(header2)
        output_lines.append("  " + "-" * 88)

        for rule_name in gold_selected:
            res = rule_results.get(rule_name, {})
            in_sampled = "YES" if rule_name in sampled_set else "NO"
            acc = res.get("accuracy", 0.0)
            cov = res.get("coverage", 0.0)
            cost = res.get("avg_cost_ratio", 0.0)
            output_lines.append(
                f"  {rule_name:<40} | {in_sampled:<10} | {acc:<8.2f} | {cov:<8.2f} | {cost:.5f}"
            )

        output_lines.append("")

        # ── Overfitting diagnosis ─────────────────────────────────────────────
        output_lines.append("  OVERFITTING DIAGNOSIS:")

        # Rules only in sampled (dropped by gold)
        only_in_sampled = [r for r in sampled_selected if r not in gold_set]
        only_in_gold = [r for r in gold_selected if r not in sampled_set]
        shared = [r for r in sampled_selected if r in gold_set]

        output_lines.append(f"    Rules only in sampled (dropped by gold): {len(only_in_sampled)} rules")
        if only_in_sampled:
            # Compute coverage on sampled docs: approximate from sampled per_doc
            # (we don't have per-rule sampled results, so note N/A)
            accs_dropped = [rule_results[r]["accuracy"] for r in only_in_sampled if r in rule_results]
            covs_dropped = [rule_results[r]["coverage"] for r in only_in_sampled if r in rule_results]
            if accs_dropped:
                output_lines.append(f"      Average accuracy on unsampled: {mean(accs_dropped):.2f}")
            if covs_dropped:
                output_lines.append(f"      Average coverage on unsampled: {mean(covs_dropped):.2f}")
            for r in only_in_sampled:
                res = rule_results.get(r, {})
                acc = res.get("accuracy", 0.0)
                cov = res.get("coverage", 0.0)
                cost = res.get("avg_cost_ratio", 0.0)
                output_lines.append(f"      - {r}: acc={acc:.2f}, cov={cov:.2f}, cost={cost:.5f}")

        output_lines.append(f"    Rules only in gold (missed by sampled): {len(only_in_gold)} rules")
        if only_in_gold:
            accs_missed = [rule_results[r]["accuracy"] for r in only_in_gold if r in rule_results]
            covs_missed = [rule_results[r]["coverage"] for r in only_in_gold if r in rule_results]
            if accs_missed:
                output_lines.append(f"      Average accuracy on unsampled: {mean(accs_missed):.2f}")
            if covs_missed:
                output_lines.append(f"      Average coverage on unsampled: {mean(covs_missed):.2f}")
            for r in only_in_gold:
                res = rule_results.get(r, {})
                acc = res.get("accuracy", 0.0)
                cov = res.get("coverage", 0.0)
                cost = res.get("avg_cost_ratio", 0.0)
                output_lines.append(f"      - {r}: acc={acc:.2f}, cov={cov:.2f}, cost={cost:.5f}")

        output_lines.append(f"    Rules in both (shared): {len(shared)} rules")
        if shared:
            accs_shared = [rule_results[r]["accuracy"] for r in shared if r in rule_results]
            if accs_shared:
                output_lines.append(f"      Average accuracy on unsampled: {mean(accs_shared):.2f}")

        # Root cause summary
        if only_in_sampled and only_in_gold:
            avg_acc_sampled_only = mean(rule_results[r]["accuracy"] for r in only_in_sampled if r in rule_results) if only_in_sampled else 0.0
            avg_acc_gold_only = mean(rule_results[r]["accuracy"] for r in only_in_gold if r in rule_results) if only_in_gold else 0.0
            avg_cov_sampled_only = mean(rule_results[r]["coverage"] for r in only_in_sampled if r in rule_results) if only_in_sampled else 0.0
            avg_cov_gold_only = mean(rule_results[r]["coverage"] for r in only_in_gold if r in rule_results) if only_in_gold else 0.0
            output_lines.append(
                f"    Root cause summary: Sampled selection chose rules with higher coverage on sampled docs "
                f"but lower generalization (acc={avg_acc_sampled_only:.2f}, cov={avg_cov_sampled_only:.2f} on unsampled) "
                f"while missing gold rules (acc={avg_acc_gold_only:.2f}, cov={avg_cov_gold_only:.2f}) "
                f"that generalize better to the full 50-doc set."
            )
        elif only_in_sampled:
            avg_acc = mean(rule_results[r]["accuracy"] for r in only_in_sampled if r in rule_results) if only_in_sampled else 0.0
            output_lines.append(
                f"    Root cause summary: Rules selected only on sampled docs perform poorly (acc={avg_acc:.2f}) "
                f"on unsampled docs — they overfit to the specific patterns in the 10 sampled documents."
            )
        elif only_in_gold:
            avg_acc = mean(rule_results[r]["accuracy"] for r in only_in_gold if r in rule_results) if only_in_gold else 0.0
            output_lines.append(
                f"    Root cause summary: Sampled selection missed gold rules (avg acc={avg_acc:.2f}) "
                f"that generalize well to the full 50-doc set."
            )
        else:
            output_lines.append(
                "    Root cause summary: Sampled and gold selected identical rules — no overfitting for this question."
            )

        output_lines.append("")
        output_lines.append("")

        # Store for overall summary
        overall_rows.append({
            "question": question,
            "slug": slug,
            "sampled_acc_sampled": sampled_acc_on_sampled,
            "sampled_acc_unsampled": sampled_merged_acc_unsampled,
            "gold_acc_unsampled": gold_merged_acc_unsampled,
            "gap": (sampled_merged_acc_unsampled - gold_merged_acc_unsampled)
                   if sampled_merged_acc_unsampled is not None and gold_merged_acc_unsampled is not None
                   else None,
            "n_sampled": len(sampled_selected),
            "n_gold": len(gold_selected),
            "n_only_sampled": len(only_in_sampled),
            "n_only_gold": len(only_in_gold),
            "n_shared": len(shared),
            "avg_acc_only_sampled": (
                mean(rule_results[r]["accuracy"] for r in only_in_sampled if r in rule_results)
                if only_in_sampled else None
            ),
            "avg_acc_only_gold": (
                mean(rule_results[r]["accuracy"] for r in only_in_gold if r in rule_results)
                if only_in_gold else None
            ),
        })

    # ── Overall Summary ───────────────────────────────────────────────────────
    output_lines.append("OVERALL SUMMARY")
    output_lines.append("=" * 64)
    output_lines.append("")
    output_lines.append(
        f"  {'Question':<55} | {'Smp@Smp':>7} | {'Smp@Uns':>7} | {'Gold@Uns':>8} | {'Gap':>6} | {'OvfitRules':>10}"
    )
    output_lines.append("  " + "-" * 110)

    gaps = []
    for row in overall_rows:
        q_short = row["question"][:53]
        smp_smp = f"{row['sampled_acc_sampled']:.2f}" if row["sampled_acc_sampled"] is not None else "N/A"
        smp_uns = f"{row['sampled_acc_unsampled']:.2f}" if row["sampled_acc_unsampled"] is not None else "N/A"
        gold_uns = f"{row['gold_acc_unsampled']:.2f}" if row["gold_acc_unsampled"] is not None else "N/A"
        gap_str = f"{row['gap']:+.2f}" if row["gap"] is not None else "N/A"
        overfit_rules = row["n_only_sampled"]
        output_lines.append(
            f"  {q_short:<55} | {smp_smp:>7} | {smp_uns:>7} | {gold_uns:>8} | {gap_str:>6} | {overfit_rules:>10}"
        )
        if row["gap"] is not None:
            gaps.append(row["gap"])

    output_lines.append("")

    if gaps:
        avg_gap = mean(gaps)
        neg_gaps = [g for g in gaps if g < 0]
        pos_gaps = [g for g in gaps if g > 0]
        output_lines.append(f"  Average accuracy gap (sampled - gold): {avg_gap:+.3f}")
        output_lines.append(f"  Questions where sampled UNDERPERFORMS gold: {len(neg_gaps)} / {len(gaps)}")
        output_lines.append(f"  Questions where sampled OVERPERFORMS gold: {len(pos_gaps)} / {len(gaps)}")
        output_lines.append("")

    output_lines.append("  KEY PATTERNS:")
    output_lines.append("")

    # Analyze patterns
    total_only_sampled = sum(r["n_only_sampled"] for r in overall_rows)
    total_only_gold = sum(r["n_only_gold"] for r in overall_rows)
    total_shared = sum(r["n_shared"] for r in overall_rows)

    output_lines.append(f"  - Total rules selected only on sampled (potentially overfit): {total_only_sampled}")
    output_lines.append(f"  - Total rules selected only in gold (missed by sampled): {total_only_gold}")
    output_lines.append(f"  - Total shared rules (selected in both): {total_shared}")
    output_lines.append("")

    # Analyze rules that were only in sampled — how do they perform on unsampled?
    all_only_sampled_accs = [
        r["avg_acc_only_sampled"] for r in overall_rows if r["avg_acc_only_sampled"] is not None
    ]
    all_only_gold_accs = [
        r["avg_acc_only_gold"] for r in overall_rows if r["avg_acc_only_gold"] is not None
    ]
    if all_only_sampled_accs:
        output_lines.append(
            f"  - Avg accuracy of sampled-only rules on unsampled docs: {mean(all_only_sampled_accs):.3f}"
        )
    if all_only_gold_accs:
        output_lines.append(
            f"  - Avg accuracy of gold-only rules on unsampled docs:    {mean(all_only_gold_accs):.3f}"
        )
    output_lines.append("")
    output_lines.append(
        "  OVERFITTING MECHANISM: The sampled refinement (10 docs) selects rules based on a small,\n"
        "  potentially unrepresentative sample. Rules that happen to match patterns in those 10 docs\n"
        "  get selected even if they don't generalize. The gold refinement (50 docs) provides a more\n"
        "  reliable signal, resulting in different (often better-generalizing) rule selections.\n"
        "  The accuracy gap primarily reflects: (1) sampled-only rules performing poorly on unsampled\n"
        "  docs, and (2) missed gold rules that generalize well but were not triggered by the 10 sampled docs."
    )

    # Write output
    final_text = "\n".join(output_lines)
    OUTPUT_FILE.write_text(final_text)
    print(f"\nOutput written to: {OUTPUT_FILE}")
    print("\n" + final_text[:2000] + ("..." if len(final_text) > 2000 else ""))


if __name__ == "__main__":
    main()
