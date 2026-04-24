"""test frozen best_rules.json on holdout documents.

Usage:
    PYTHONPATH=src python -m agent.rule_runtime.holdout \
        --config src/agent/config_pdfs_10doc.yaml \
        --packaging-mode full_bundle_reference \
        --queries 1,3,4,8,9 \
        --max-docs 15 \
        [--dry-run] [--query-only 1]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, TypedDict

import yaml

from agent.rule_runtime.artifacts import (
    collect_sampled_doc_ids_from_best_rules,
    load_best_rules_payload,
    rule_from_best_rules_entry,
)
from agent.rule_runtime.data import (
    DocumentSample,
    estimate_tokens,
    extract_ground_truth,
    get_label_filename,
    get_query_text,
    reconstruct_to_normalized_text,
)
from agent.rules.range_rule_exec import RetrievedSpan, execute_range_rule
from agent.rules.range_rule_json import RangeRule
from agent.rules.range_rule_scorer import score_retrieved_subset
from core.pipeline.e2e_utils.cache import CachedLLMCaller, DEFAULT_CACHE_DB_PATH

_INFO_NOT_FOUND = "information not found."
_DEFAULT_RETRIEVAL_TOO_LARGE_TOKEN_THRESHOLD = 5000
_DEFAULT_OUTPUT_ROOT = Path("output/agent/holdout")

# Hint gate threshold: when rule.phase_a_hint_reliability >= this value and the
# hint regex does not match, the pair is treated as judge=False (skipping LLM
# gen+judge). Conservative at 0.8 — the hint must explain >=80% of judge_pass
# cases on the sampled set to be trusted. Lower-reliability hints bypass the gate
# entirely (zero regression).
_HINT_GATE_RELIABILITY_THRESHOLD = 0.8

# Skip-gen threshold (stricter than the gate, because it replaces LLM generation
# entirely). When hint reliability >= this value, extraction precision >= threshold,
# the regex matches, and the span is short, the regex capture group is used as the
# candidate answer and only the judge is called. This eliminates the generation
# call, which is the dominant cost (~80% per pair). Short spans = high-precision
# case, so the risk is manageable.
_SKIP_GEN_RELIABILITY_THRESHOLD = 0.9
# Extraction precision gate: skip-gen is only safe when the first regex match is
# likely the actual answer. Empirically, a capitalized-word pattern can have
# reliability=1.0 but extraction precision=0 (match='Item', GT='Delaware'),
# causing 100% mis-judgment without this guard.
_SKIP_GEN_EXTRACTION_PRECISION_THRESHOLD = 0.8
_SKIP_GEN_MAX_SPAN_CHARS = 800


class HoldoutEvalRow(TypedDict):
    """Strict subset of runner eval_rows, containing only holdout-relevant fields."""

    query_idx: int
    query_text: str
    doc_id: str
    rule_index: int
    rule_text: str
    retrieval_spec: dict[str, Any]
    matched: bool
    retrieved_subset_text: str
    retrieved_subset_chars: int
    retrieved_subset_tokens: int
    ground_truth: str
    generated_answer: str | None
    judge_result: bool | str  # True/False or "NOT_RUN"
    blocker: str | None
    actual_cost_usd: float
    eval_mode: str  # "per_rule" or "union"
    packaging_mode: str


def load_experiment_config(config_path: Path) -> dict[str, Any]:
    """Load experiment config from YAML."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_frozen_rules(
    query_idx: int,
    packaging_mode: str,
    output_root: Path = _DEFAULT_OUTPUT_ROOT,
) -> tuple[list[RangeRule], set[str]]:
    """Load frozen rules from best_rules.json and extract sampled doc IDs.

    Returns:
        (rules, sampled_doc_ids): rule list + set of doc IDs used during rule generation
    """
    path = output_root / f"q{query_idx}" / packaging_mode / "best_rules.json"
    data = load_best_rules_payload(path)
    sampled_summary = load_sampled_summary(query_idx, packaging_mode, output_root)

    rules: list[RangeRule] = []

    for mr_dict in data["merged_rules"]:
        rules.append(rule_from_best_rules_entry(mr_dict))

    sampled_doc_ids = collect_sampled_doc_ids_from_best_rules(data, sampled_summary)

    return rules, sampled_doc_ids


def load_sampled_summary(
    query_idx: int,
    packaging_mode: str,
    output_root: Path = _DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any] | None:
    """Load the sampled run's summary.json (used for generalization gap comparison)."""
    path = output_root / f"q{query_idx}" / packaging_mode / "summary.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_holdout_docs(
    label_path: Path,
    query_idx: int,
    processing_dir: Path,
    sampled_doc_ids: set[str],
    max_docs: int = 15,
    seed: int = 42,
) -> list[str]:
    """Select holdout documents using seeded random sampling.

    Filters:
    1. Not in sampled_doc_ids
    2. possible_provenance_nodes is non-empty
    3. Corresponding _reconstructed.json exists

    Candidates are sorted before sampling for stable input order. The sampled
    result is sorted afterwards for stable downstream evaluation order.
    """
    with label_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    candidates: list[str] = []
    for entry in data.get("labels", []):
        doc_id = entry.get("doc_name", "")
        if doc_id in sampled_doc_ids:
            continue
        prov = entry.get("possible_provenance_nodes", [])
        if not prov:
            continue
        recon_path = processing_dir / f"{doc_id}_reconstructed.json"
        if not recon_path.exists():
            continue
        candidates.append(doc_id)

    candidates.sort()
    chosen = _random_sample(candidates, max_docs, seed)
    return sorted(chosen)


def _random_sample(doc_ids: list[str], max_docs: int, seed: int) -> list[str]:
    """Reproducible unbiased sample using random.Random(seed); returns all candidates if fewer than max_docs.

    max_docs == 0 means "unlimited"; negatives are rejected to catch CLI typos
    that would otherwise silently return the full list.
    """
    if max_docs < 0:
        raise ValueError(f"max_docs must be >= 0 (0 = unlimited), got {max_docs}")
    if max_docs == 0 or max_docs >= len(doc_ids):
        return list(doc_ids)
    rng = random.Random(seed)
    return rng.sample(doc_ids, max_docs)


def build_holdout_docs(
    doc_ids: list[str],
    query_idx: int,
    processing_dir: Path,
    label_path: Path,
    truncate_before: str | None,
) -> list[DocumentSample]:
    """Build a list of DocumentSample objects for the given doc IDs."""
    docs: list[DocumentSample] = []
    for doc_id in doc_ids:
        recon_path = processing_dir / f"{doc_id}_reconstructed.json"
        text = reconstruct_to_normalized_text(recon_path, truncate_before=truncate_before)
        gt = extract_ground_truth(label_path, doc_id, query_idx)
        token_count = estimate_tokens(text)
        docs.append(
            DocumentSample(
                doc_id=doc_id,
                markdown_text=text,
                ground_truth_answer=gt,
                token_count=token_count,
            )
        )
    return docs


def _is_information_not_found(answer: str | None) -> bool:
    if answer is None:
        return False
    return answer.strip().casefold() == _INFO_NOT_FOUND


def classify_blocker(
    matched: bool,
    subset_tokens: int,
    generated_answer: str | None,
    judge_result: Any,
    threshold: int,
    metadata: dict[str, Any],
) -> str | None:
    """Classify the failure reason for a (rule, doc) evaluation attempt."""
    if not matched:
        return str(metadata.get("reason", "retrieval_unmatched"))
    if subset_tokens <= 0:
        return "empty_span"
    if judge_result is True:
        return None
    if subset_tokens >= threshold:
        return "retrieval_too_large"
    if _is_information_not_found(generated_answer):
        return "gen_not_found"
    return "judge_false"


def evaluate_rule_on_doc(
    rule: RangeRule,
    rule_index: int,
    doc: DocumentSample,
    query_idx: int,
    query_text: str,
    cached_caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
    retrieval_too_large_token_threshold: int,
) -> HoldoutEvalRow:
    """Run the extract -> gen -> judge pipeline for a single (rule, doc) pair."""
    subset = execute_range_rule(rule, doc.markdown_text)
    subset_text = "\n\n".join(span.text for span in subset.spans) if subset.spans else ""
    subset_chars = len(subset_text)
    subset_tokens = estimate_tokens(subset_text) if subset_text else 0

    row: dict[str, Any] = {
        "query_idx": query_idx,
        "query_text": query_text,
        "doc_id": doc.doc_id,
        "rule_index": rule_index,
        "rule_text": rule.rule_text,
        "retrieval_spec": rule.retrieval_spec.to_dict(),
        "matched": subset.matched,
        "retrieved_subset_text": subset_text,
        "retrieved_subset_chars": subset_chars,
        "retrieved_subset_tokens": subset_tokens,
        "ground_truth": doc.ground_truth_answer,
        "generated_answer": None,
        "judge_result": "NOT_RUN",
        "blocker": None,
        "actual_cost_usd": 0.0,
        "eval_mode": "per_rule",
        "packaging_mode": "",  # filled by caller
    }

    # Not matched: skip gen+judge
    if not subset.matched:
        row["blocker"] = classify_blocker(
            False, 0, None, "NOT_RUN", retrieval_too_large_token_threshold, subset.metadata,
        )
        return row  # type: ignore[return-value]

    # Over token threshold or empty span: classify via classify_blocker
    if subset_tokens >= retrieval_too_large_token_threshold or subset_tokens <= 0:
        row["blocker"] = classify_blocker(
            True, subset_tokens, None, False, retrieval_too_large_token_threshold, subset.metadata,
        )
        return row  # type: ignore[return-value]

    # 0-cost hint gate: when the rule carries a hint pattern and sampled reliability
    # is high enough, a regex miss directly yields judge=False (skipping LLM gen+judge).
    # Conservative design: only enabled at reliability >= threshold; lower-reliability
    # hints bypass the gate and keep the original gen+judge path (zero regression).
    hint_pattern = rule.answer_hint_pattern
    hint_reliability = rule.phase_a_hint_reliability
    hint_match: re.Match[str] | None = None
    if (
        hint_pattern
        and hint_reliability is not None
        and hint_reliability >= _HINT_GATE_RELIABILITY_THRESHOLD
    ):
        try:
            hint_match = re.search(hint_pattern, subset_text)
            if not hint_match:
                row["judge_result"] = False
                row["blocker"] = "hint_absent_skipped"
                return row  # type: ignore[return-value]
        except re.error:
            # Malformed regex (should have been validated in Phase A); fall back to gen+judge
            hint_match = None

    # Skip-gen mode: when hint reliably matches AND extraction precision is high AND
    # the span is short, use the regex capture as the candidate answer and call only
    # the judge. Generation is the dominant cost (~80% per pair for span-sized input);
    # skipping it is safe for short, high-precision spans (<= 800 chars).
    # Extraction precision is the critical guard: a pattern with reliability=1.0 can
    # still have extraction precision=0 if the first match is not the answer.
    extraction_precision = rule.phase_a_hint_extraction_precision
    if (
        hint_match is not None
        and hint_reliability is not None
        and hint_reliability >= _SKIP_GEN_RELIABILITY_THRESHOLD
        and extraction_precision is not None
        and extraction_precision >= _SKIP_GEN_EXTRACTION_PRECISION_THRESHOLD
        and len(subset_text) <= _SKIP_GEN_MAX_SPAN_CHARS
    ):
        from agent.rules.range_rule_scorer import score_generated_answer
        candidate = hint_match.group(0)
        skip_score = score_generated_answer(
            question=query_text,
            generated_answer=candidate,
            ground_truth=doc.ground_truth_answer,
            cached_caller=cached_caller,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        row["generated_answer"] = candidate
        row["judge_result"] = skip_score.judge_result
        judge_cost = skip_score.metadata.get("judge", {}).get("cost_usd", 0.0)
        row["actual_cost_usd"] = judge_cost
        row["blocker"] = (
            "skip_gen_judged_false"
            if not skip_score.judge_result
            else None
        )
        row["eval_mode"] = "per_rule_skip_gen"
        return row  # type: ignore[return-value]

    # gen → judge
    score = score_retrieved_subset(
        question=query_text,
        retrieved_text=subset_text,
        ground_truth=doc.ground_truth_answer,
        cached_caller=cached_caller,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    row["generated_answer"] = score.generated_answer
    row["judge_result"] = score.judge_result
    gen_cost = score.metadata.get("generation", {}).get("cost_usd", 0.0)
    judge_cost = score.metadata.get("judge", {}).get("cost_usd", 0.0)
    row["actual_cost_usd"] = gen_cost + judge_cost
    row["blocker"] = classify_blocker(
        True, subset_tokens, score.generated_answer, score.judge_result,
        retrieval_too_large_token_threshold, subset.metadata,
    )
    return row  # type: ignore[return-value]


def run_per_rule_eval(
    query_idx: int,
    query_text: str,
    rules: list[RangeRule],
    docs: list[DocumentSample],
    cached_caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
    retrieval_too_large_token_threshold: int,
    packaging_mode: str = "",
) -> list[HoldoutEvalRow]:
    """Evaluate the full rule x doc matrix."""
    rows: list[HoldoutEvalRow] = []
    total = len(rules) * len(docs)
    for ri, rule in enumerate(rules):
        for di, doc in enumerate(docs):
            idx = ri * len(docs) + di + 1
            row = evaluate_rule_on_doc(
                rule, ri, doc, query_idx, query_text,
                cached_caller, llm_provider, llm_model,
                retrieval_too_large_token_threshold,
            )
            row["packaging_mode"] = packaging_mode
            status = "✓" if row["judge_result"] is True else ("✗" if row["matched"] else "—")
            print(f"  [q{query_idx}] Rule {ri+1}/{len(rules)}, "
                  f"Doc {di+1}/{len(docs)} ({idx}/{total}): "
                  f"{doc.doc_id} → {status}")
            rows.append(row)
    return rows


def merge_overlapping_spans(
    spans: list[tuple[int, int]],
    document_text: str,
) -> list[RetrievedSpan]:
    """Sorted interval merge; returns a list of non-overlapping RetrievedSpan objects."""
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda s: s[0])
    merged: list[tuple[int, int]] = [sorted_spans[0]]
    for start, end in sorted_spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    result: list[RetrievedSpan] = []
    for start, end in merged:
        text = document_text[start:end]
        if text.strip():
            result.append(RetrievedSpan(start=start, end=end, text=text))
    return result


def evaluate_union_on_doc(
    rules: list[RangeRule],
    doc: DocumentSample,
    query_idx: int,
    query_text: str,
    cached_caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
    retrieval_too_large_token_threshold: int,
) -> HoldoutEvalRow:
    """Merge matched spans from all rules and run a single gen+judge pass."""
    all_span_bounds: list[tuple[int, int]] = []
    matched_rules: list[int] = []

    for ri, rule in enumerate(rules):
        subset = execute_range_rule(rule, doc.markdown_text)
        if subset.matched and subset.spans:
            matched_rules.append(ri)
            for span in subset.spans:
                all_span_bounds.append((span.start, span.end))

    row: dict[str, Any] = {
        "query_idx": query_idx,
        "query_text": query_text,
        "doc_id": doc.doc_id,
        "rule_index": -1,  # no single rule in union mode
        "rule_text": f"union({len(matched_rules)}/{len(rules)} matched)",
        "retrieval_spec": {"mode": "union", "matched_rules": matched_rules},
        "matched": bool(matched_rules),
        "retrieved_subset_text": "",
        "retrieved_subset_chars": 0,
        "retrieved_subset_tokens": 0,
        "ground_truth": doc.ground_truth_answer,
        "generated_answer": None,
        "judge_result": "NOT_RUN",
        "blocker": None,
        "actual_cost_usd": 0.0,
        "eval_mode": "union",
        "packaging_mode": "",  # filled by caller
    }

    if not matched_rules:
        row["blocker"] = "all_rules_unmatched"
        return row  # type: ignore[return-value]

    merged_spans = merge_overlapping_spans(all_span_bounds, doc.markdown_text)
    union_text = "\n\n---\n\n".join(s.text for s in merged_spans)
    union_chars = len(union_text)
    union_tokens = estimate_tokens(union_text) if union_text else 0

    row["retrieved_subset_chars"] = union_chars
    row["retrieved_subset_tokens"] = union_tokens
    row["retrieved_subset_text"] = union_text

    if union_tokens >= retrieval_too_large_token_threshold:
        row["blocker"] = "retrieval_too_large"
        return row  # type: ignore[return-value]

    if union_tokens <= 0:
        row["blocker"] = "empty_span"
        return row  # type: ignore[return-value]

    score = score_retrieved_subset(
        question=query_text,
        retrieved_text=union_text,
        ground_truth=doc.ground_truth_answer,
        cached_caller=cached_caller,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    row["generated_answer"] = score.generated_answer
    row["judge_result"] = score.judge_result
    gen_cost = score.metadata.get("generation", {}).get("cost_usd", 0.0)
    judge_cost = score.metadata.get("judge", {}).get("cost_usd", 0.0)
    row["actual_cost_usd"] = gen_cost + judge_cost
    row["blocker"] = classify_blocker(
        True, union_tokens, score.generated_answer, score.judge_result,
        retrieval_too_large_token_threshold, {},
    )
    return row  # type: ignore[return-value]


def run_union_eval(
    query_idx: int,
    query_text: str,
    rules: list[RangeRule],
    docs: list[DocumentSample],
    cached_caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
    retrieval_too_large_token_threshold: int,
    packaging_mode: str = "",
) -> list[HoldoutEvalRow]:
    """Run union evaluation for each doc."""
    rows: list[HoldoutEvalRow] = []
    for di, doc in enumerate(docs):
        row = evaluate_union_on_doc(
            rules, doc, query_idx, query_text,
            cached_caller, llm_provider, llm_model,
            retrieval_too_large_token_threshold,
        )
        row["packaging_mode"] = packaging_mode
        status = "✓" if row["judge_result"] is True else ("✗" if row["matched"] else "—")
        print(f"  [q{query_idx}] Union Doc {di+1}/{len(docs)}: {doc.doc_id} → {status}")
        rows.append(row)
    return rows


def build_per_rule_summary(
    rules: list[RangeRule],
    rows: list[HoldoutEvalRow],
) -> list[dict[str, Any]]:
    """Compute per-rule aggregate statistics."""
    summaries: list[dict[str, Any]] = []
    for ri, rule in enumerate(rules):
        rule_rows = [r for r in rows if r["rule_index"] == ri]
        total = len(rule_rows)
        if total == 0:
            continue
        matched = sum(1 for r in rule_rows if r["matched"])
        successes = sum(1 for r in rule_rows if r["judge_result"] is True)
        cost = sum(r["actual_cost_usd"] for r in rule_rows)

        # Blocker distribution
        blocker_counts: dict[str, int] = {}
        for r in rule_rows:
            b = r.get("blocker")
            if b is not None:
                blocker_counts[b] = blocker_counts.get(b, 0) + 1

        summaries.append({
            "rule_index": ri,
            "rule_text": rule.rule_text[:80],
            "retrieval_spec": rule.retrieval_spec.to_dict(),
            "holdout_accuracy": successes / total if total else 0.0,
            "holdout_coverage": matched / total if total else 0.0,
            "total_docs": total,
            "matched_docs": matched,
            "success_docs": successes,
            "total_cost_usd": round(cost, 6),
            "avg_cost_usd": round(cost / total, 6) if total else 0.0,
            "blocker_breakdown": blocker_counts,
        })
    return summaries


def build_holdout_report(
    query_idx: int,
    query_text: str,
    per_rule_rows: list[HoldoutEvalRow],
    union_rows: list[HoldoutEvalRow],
    rules: list[RangeRule],
    sampled_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the complete holdout evaluation report."""
    per_rule_summary = build_per_rule_summary(rules, per_rule_rows)

    # Global per-rule metrics
    per_rule_cost = sum(r["actual_cost_usd"] for r in per_rule_rows)
    best_acc = max((s["holdout_accuracy"] for s in per_rule_summary), default=0.0)
    avg_acc = (
        sum(s["holdout_accuracy"] for s in per_rule_summary) / len(per_rule_summary)
        if per_rule_summary else 0.0
    )

    # Union metrics
    union_total = len(union_rows)
    union_successes = sum(1 for r in union_rows if r["judge_result"] is True)
    union_cost = sum(r["actual_cost_usd"] for r in union_rows)
    union_coverage = sum(1 for r in union_rows if r["matched"]) / union_total if union_total else 0.0
    union_accuracy = union_successes / union_total if union_total else 0.0

    # Generalization gap
    sampled_best_acc = None
    generalization_gap = None
    if sampled_summary:
        sampled_best_acc = sampled_summary.get("best_single_rule_accuracy")
        if sampled_best_acc is not None:
            generalization_gap = round(best_acc - sampled_best_acc, 4)

    # Failure details
    failure_details: list[dict[str, Any]] = []
    for r in per_rule_rows:
        if r["judge_result"] is not True and r["blocker"] is not None:
            failure_details.append({
                "doc_id": r["doc_id"],
                "rule_index": r["rule_index"],
                "rule_text": r["rule_text"][:80],
                "blocker": r["blocker"],
                "matched": r["matched"],
                "retrieved_subset_chars": r["retrieved_subset_chars"],
                "generated_answer": (r["generated_answer"] or "")[:200],
                "ground_truth": r["ground_truth"][:200],
            })

    # Global blocker distribution
    global_blockers: dict[str, int] = {}
    for r in per_rule_rows:
        b = r.get("blocker")
        if b is not None:
            global_blockers[b] = global_blockers.get(b, 0) + 1

    return {
        "query_idx": query_idx,
        "query_text": query_text,
        "holdout_doc_count": union_total,
        "rule_count": len(rules),
        "per_rule_summary": per_rule_summary,
        "best_acc": round(best_acc, 4),
        "avg_acc": round(avg_acc, 4),
        "per_rule_total_cost_usd": round(per_rule_cost, 4),
        "union_summary": {
            "accuracy": round(union_accuracy, 4),
            "coverage": round(union_coverage, 4),
            "total_cost_usd": round(union_cost, 4),
            "success_docs": union_successes,
            "total_docs": union_total,
        },
        "sampled_best_acc": sampled_best_acc,
        "generalization_gap": generalization_gap,
        "failure_breakdown": global_blockers,
        "failure_details": failure_details,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render the holdout evaluation report as a Markdown string."""
    lines: list[str] = []
    q = report["query_idx"]
    lines.append(f"# Holdout Evaluation: q{q}")
    lines.append(f"\n**Query**: {report['query_text']}")
    lines.append(f"**Holdout docs**: {report['holdout_doc_count']}")
    lines.append(f"**Rules**: {report['rule_count']}")

    # Generalization gap
    if report["sampled_best_acc"] is not None:
        gap = report["generalization_gap"]
        gap_str = f"{gap:+.1%}" if gap is not None else "N/A"
        lines.append(
            f"\n**Sampled BestAcc**: {report['sampled_best_acc']:.1%} → "
            f"**Holdout BestAcc**: {report['best_acc']:.1%} "
            f"(gap: {gap_str})"
        )

    # Per-rule table
    lines.append("\n## Per-Rule Results\n")
    lines.append("| Rule | Accuracy | Coverage | AvgCost | TopBlocker |")
    lines.append("|------|----------|----------|---------|------------|")
    for s in report["per_rule_summary"]:
        top_blocker = max(s["blocker_breakdown"], key=s["blocker_breakdown"].get) if s["blocker_breakdown"] else "—"
        lines.append(
            f"| R{s['rule_index']} | {s['holdout_accuracy']:.1%} "
            f"| {s['holdout_coverage']:.1%} "
            f"| ${s['avg_cost_usd']:.4f} "
            f"| {top_blocker} |"
        )

    # Summary row
    lines.append(
        f"\n**BestAcc**: {report['best_acc']:.1%} | "
        f"**AvgAcc**: {report['avg_acc']:.1%} | "
        f"**PerRuleCost**: ${report['per_rule_total_cost_usd']:.4f}"
    )

    # Union results
    u = report["union_summary"]
    lines.append("\n## Union Results\n")
    lines.append("| Accuracy | Coverage | TotalCost |")
    lines.append("|----------|----------|-----------|")
    lines.append(f"| {u['accuracy']:.1%} | {u['coverage']:.1%} | ${u['total_cost_usd']:.4f} |")

    # Blocker distribution
    if report["failure_breakdown"]:
        lines.append("\n## Failure Breakdown\n")
        lines.append("| Blocker | Count |")
        lines.append("|---------|-------|")
        for blocker, count in sorted(report["failure_breakdown"].items(), key=lambda x: -x[1]):
            lines.append(f"| {blocker} | {count} |")

    return "\n".join(lines)


def save_results(
    output_dir: Path,
    per_rule_rows: list[HoldoutEvalRow],
    union_rows: list[HoldoutEvalRow],
    report: dict[str, Any],
) -> None:
    """Write holdout evaluation results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "holdout_eval_rows.jsonl").open("w", encoding="utf-8") as f:
        for row in per_rule_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (output_dir / "holdout_union_rows.jsonl").open("w", encoding="utf-8") as f:
        for row in union_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (output_dir / "holdout_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md = render_markdown_report(report)
    with (output_dir / "holdout_report.md").open("w", encoding="utf-8") as f:
        f.write(md)


def save_aggregate_report(
    output_root: Path,
    all_reports: dict[int, dict[str, Any]],
) -> None:
    """Write the cross-query aggregate report."""
    with (output_root / "holdout_aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)

    lines: list[str] = ["# Holdout Evaluation Aggregate\n"]
    lines.append("| Query | Docs | Rules | BestAcc | AvgAcc | UnionAcc | PerRuleCost | UnionCost | Gap |")
    lines.append("|-------|------|-------|---------|--------|----------|-------------|-----------|-----|")
    for qi in sorted(all_reports.keys()):
        r = all_reports[qi]
        u = r["union_summary"]
        gap = r.get("generalization_gap")
        gap_str = f"{gap:+.1%}" if gap is not None else "—"
        lines.append(
            f"| q{qi} | {r['holdout_doc_count']} | {r['rule_count']} "
            f"| {r['best_acc']:.1%} | {r['avg_acc']:.1%} "
            f"| {u['accuracy']:.1%} "
            f"| ${r['per_rule_total_cost_usd']:.4f} "
            f"| ${u['total_cost_usd']:.4f} "
            f"| {gap_str} |"
        )

    with (output_root / "holdout_aggregate.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H1 rule holdout generalization evaluation")
    parser.add_argument("--config", type=Path, required=True, help="Experiment config YAML")
    parser.add_argument("--packaging-mode", default="full_bundle_reference")
    parser.add_argument("--queries", default="1,3,4,8,9", help="Comma-separated query indices")
    parser.add_argument("--max-docs", type=int, default=15, help="Max holdout docs per query (0=unlimited)")
    parser.add_argument("--holdout-seed", type=int, default=42, help="Seeded random holdout selection seed")
    parser.add_argument("--dry-run", action="store_true", help="Print doc list only, no LLM calls")
    parser.add_argument("--query-only", type=int, default=None, help="Run a single query only")
    parser.add_argument("--llm-provider", default=None, help="Override llm_provider from config")
    parser.add_argument("--llm-model", default=None, help="Override llm_model from config")
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)

    dataset_root = config.get("dataset_root", "datasets/pdfs/latest")
    # Derive directory suffix from parser field (consistent with tool_agent/cli.py)
    parser = config.get("parser", "docling")
    if parser == "mineru":
        processing_dir = Path(dataset_root) / "processing_mineru"
        label_dir = Path(dataset_root) / "label_mineru"
    elif parser == "docling":
        processing_dir = Path(dataset_root) / "processing"
        label_dir = Path(dataset_root) / "label"
    else:
        raise ValueError(f"Unsupported parser: {parser!r}; expected 'docling' or 'mineru'")
    truncate_before = config.get("truncate_before")
    llm_provider = args.llm_provider or config.get("llm_provider", "azure")
    llm_model = args.llm_model or config.get("llm_model")
    if not llm_model:
        raise ValueError("llm_model must be specified in config or --llm-model")
    threshold = config.get(
        "retrieval_too_large_token_threshold",
        _DEFAULT_RETRIEVAL_TOO_LARGE_TOKEN_THRESHOLD,
    )

    query_indices = [int(q) for q in args.queries.split(",")]
    if args.query_only is not None:
        query_indices = [args.query_only]

    print("=== H1 Holdout Evaluation ===")
    print(f"Config: {args.config}")
    print(f"Mode: {args.packaging_mode}")
    print(f"Queries: {query_indices}")
    print(f"Max docs/query: {args.max_docs}")
    print(f"Holdout seed: {args.holdout_seed}")
    print(f"LLM: {llm_provider}/{llm_model}")
    print(f"Truncate before: {truncate_before}")
    print()

    # Shared LLM cache across pipelines
    cached_caller = CachedLLMCaller(DEFAULT_CACHE_DB_PATH)

    all_reports: dict[int, dict[str, Any]] = {}

    for qi in query_indices:
        print(f"--- Query q{qi} ---")
        t0 = time.time()

        # Load frozen rules and sampled doc IDs
        rules, sampled_doc_ids = load_frozen_rules(qi, args.packaging_mode, args.output_root)
        sampled_summary = load_sampled_summary(qi, args.packaging_mode, args.output_root)

        # Select holdout docs
        label_path = label_dir / get_label_filename(
            {"dataset": config.get("dataset", "pdfs")}, qi
        )
        holdout_ids = select_holdout_docs(
            label_path, qi, processing_dir, sampled_doc_ids, args.max_docs,
            seed=args.holdout_seed,
        )
        print(f"  Rules: {len(rules)}, Sampled docs: {len(sampled_doc_ids)}, "
              f"Holdout docs: {len(holdout_ids)}")
        print(f"  Holdout: {holdout_ids}")

        if args.dry_run:
            per_rule_calls = len(rules) * len(holdout_ids) * 2  # gen + judge
            union_calls = len(holdout_ids) * 2
            print(f"  [DRY-RUN] Est. LLM calls: {per_rule_calls} per-rule + {union_calls} union")
            print()
            continue

        # Build document samples
        docs = build_holdout_docs(holdout_ids, qi, processing_dir, label_path, truncate_before)
        print(f"  Loaded {len(docs)} docs, total tokens: {sum(d.token_count for d in docs)}")

        print("\n  === Per-Rule Evaluation ===")
        per_rule_rows = run_per_rule_eval(
            qi, get_query_text(dataset_root, qi), rules, docs,
            cached_caller, llm_provider, llm_model, threshold,
            packaging_mode=args.packaging_mode,
        )

        print("\n  === Union Evaluation ===")
        union_rows = run_union_eval(
            qi, get_query_text(dataset_root, qi), rules, docs,
            cached_caller, llm_provider, llm_model, threshold,
            packaging_mode=args.packaging_mode,
        )

        report = build_holdout_report(
            qi, get_query_text(dataset_root, qi),
            per_rule_rows, union_rows, rules, sampled_summary,
        )
        all_reports[qi] = report

        out_dir = args.output_root / f"q{qi}" / args.packaging_mode / "holdout"
        save_results(out_dir, per_rule_rows, union_rows, report)

        elapsed = time.time() - t0
        print(f"\n  q{qi} done in {elapsed:.1f}s — BestAcc: {report['best_acc']:.1%}, "
              f"UnionAcc: {report['union_summary']['accuracy']:.1%}")
        print()

    if all_reports:
        save_aggregate_report(args.output_root, all_reports)
        print("=== Aggregate Report ===")
        for qi in sorted(all_reports.keys()):
            r = all_reports[qi]
            gap = r.get("generalization_gap")
            gap_str = f"{gap:+.1%}" if gap is not None else "—"
            print(f"  q{qi}: BestAcc={r['best_acc']:.1%}, "
                  f"UnionAcc={r['union_summary']['accuracy']:.1%}, "
                  f"Gap={gap_str}")
        print(f"\nResults saved to {args.output_root}/")


if __name__ == "__main__":
    main()
