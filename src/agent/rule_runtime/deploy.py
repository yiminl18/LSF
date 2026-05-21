"""Deployable rule policies for unsampled holdout documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

import yaml

from agent.rule_runtime.artifacts import (
    collect_sampled_doc_ids_from_best_rules,
    load_best_rules_payload,
    load_rules_from_best_rules,
    load_sampled_eval_from_best_rules,
    rank_rules_by_sampled_acc,
)
from agent.rule_runtime import rule_dispatch
from agent.rule_runtime.data import (
    DocumentSample,
    estimate_tokens,
    get_label_filename,
    get_query_text,
)
from agent.rule_runtime.holdout import build_holdout_docs, select_holdout_docs
from agent.rules.range_rule_scorer import (
    generate_answer_from_text,
    score_generated_answer,
)
from core.pipeline.e2e_utils.cache import CachedLLMCaller, DEFAULT_CACHE_DB_PATH


_INFO_NOT_FOUND_NORMALIZED = "information not found."
_POLICY_NAME = "cascade"


class DeployedRow(TypedDict):
    """Per-doc deployment outcome row."""

    query_idx: int
    doc_id: str
    policy: str
    rules_used: list[int]
    retrieved_subset_text: str
    generated_answer: str | None
    judge_result: bool | str
    blocker: str | None
    gen_calls: int
    judge_calls: int
    # actual_cost_usd is gen_cost_usd + judge_cost_usd. The split fields let
    # callers compare reader spend against judge spend without re-parsing
    # ExtractionResult.trace.
    actual_cost_usd: float
    gen_cost_usd: float
    judge_cost_usd: float
    latency_ms: float


def _empty_row(
    query_idx: int,
    doc_id: str,
    policy: str,
    blocker: str,
    rules_used: list[int] | None = None,
) -> DeployedRow:
    return DeployedRow(
        query_idx=query_idx,
        doc_id=doc_id,
        policy=policy,
        rules_used=list(rules_used or []),
        retrieved_subset_text="",
        generated_answer=None,
        judge_result=False,
        blocker=blocker,
        gen_calls=0,
        judge_calls=0,
        actual_cost_usd=0.0,
        gen_cost_usd=0.0,
        judge_cost_usd=0.0,
        latency_ms=0.0,
    )


def summarize_rows(rows: list[DeployedRow], policy_name: str = _POLICY_NAME) -> dict[str, Any]:
    """Aggregate DeployedRow list into a deployed_summary.json payload."""
    n = len(rows)
    if n == 0:
        return {
            "policy": policy_name,
            "n_docs": 0,
            "deployed_acc": 0.0,
            "total_cost_usd": 0.0,
            "total_gen_cost_usd": 0.0,
            "total_judge_cost_usd": 0.0,
            "n_judge_pass": 0,
            "mean_gen_calls": 0.0,
            "mean_judge_calls": 0.0,
            "p50_gen_calls": 0.0,
            "p95_gen_calls": 0.0,
            "total_latency_ms": 0.0,
            "mean_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }
    n_pass = sum(1 for r in rows if r["judge_result"] is True)
    total_cost = sum(r["actual_cost_usd"] for r in rows)
    # gen/judge cost split fields are recent additions (see DeployedRow).
    # Legacy rows from older runs lack them; treat missing as 0.0 so this
    # function still produces a payload when applied to mixed data.
    gen_cost = sum(float(r.get("gen_cost_usd", 0.0) or 0.0) for r in rows)
    judge_cost = sum(float(r.get("judge_cost_usd", 0.0) or 0.0) for r in rows)
    gen_calls = sorted(r["gen_calls"] for r in rows)
    judge_calls = sorted(r["judge_calls"] for r in rows)
    latencies = sorted(float(r.get("latency_ms", 0.0) or 0.0) for r in rows)

    def _percentile(sorted_xs: list[float], p: float) -> float:
        if not sorted_xs:
            return 0.0
        k = max(0, min(len(sorted_xs) - 1, int(round(p * (len(sorted_xs) - 1)))))
        return float(sorted_xs[k])

    return {
        "policy": policy_name,
        "n_docs": n,
        "n_judge_pass": n_pass,
        "deployed_acc": round(n_pass / n, 4),
        "total_cost_usd": round(total_cost, 6),
        "total_gen_cost_usd": round(gen_cost, 6),
        "total_judge_cost_usd": round(judge_cost, 6),
        "mean_gen_calls": round(sum(gen_calls) / n, 3),
        "mean_judge_calls": round(sum(judge_calls) / n, 3),
        "p50_gen_calls": _percentile(gen_calls, 0.5),
        "p95_gen_calls": _percentile(gen_calls, 0.95),
        "total_latency_ms": round(sum(latencies), 3),
        "mean_latency_ms": round(sum(latencies) / n, 3),
        "p50_latency_ms": round(_percentile(latencies, 0.5), 3),
        "p95_latency_ms": round(_percentile(latencies, 0.95), 3),
    }


def _is_information_not_found(answer: str | None) -> bool:
    if not answer:
        return False
    return answer.strip().casefold() == _INFO_NOT_FOUND_NORMALIZED


def _retrieve(
    rule: rule_dispatch.Rule,
    doc: DocumentSample,
) -> tuple[bool, str, int, str | None]:
    """Retrieve span. Returns (usable, span_text, tokens, blocker_or_None)."""
    subset = rule_dispatch.apply_rule(rule, doc.markdown_text)
    if not subset.matched:
        return False, "", 0, str(subset.metadata.get("reason", "retrieval_unmatched"))
    span_text = "\n\n".join(s.text for s in subset.spans) if subset.spans else ""
    if not span_text.strip():
        return False, "", 0, "empty_span"
    return True, span_text, estimate_tokens(span_text), None


def _judge_only(
    query_text: str,
    candidate_answer: str,
    ground_truth: str,
    cached_caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
) -> tuple[bool, float]:
    score = score_generated_answer(
        question=query_text,
        generated_answer=candidate_answer,
        ground_truth=ground_truth,
        cached_caller=cached_caller,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    judge_cost = score.metadata.get("judge", {}).get("cost_usd", 0.0)
    return bool(score.judge_result), judge_cost


def evaluate_cascade(
    rules: list[rule_dispatch.Rule],
    sampled_eval: dict[int, dict[str, Any]],
    holdout_docs: list[DocumentSample],
    query_idx: int,
    query_text: str,
    cached_caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
    retrieval_too_large_token_threshold: int,
) -> list[DeployedRow]:
    """Cascade over rules; gen-only per attempt, judge once on final answer."""
    if not rules:
        return [
            _empty_row(query_idx, d.doc_id, _POLICY_NAME, "no_rules")
            for d in holdout_docs
        ]
    ranked = rank_rules_by_sampled_acc(rules, sampled_eval)
    rows: list[DeployedRow] = []
    for doc in holdout_docs:
        rules_tried: list[int] = []
        gen_calls = 0
        cum_cost = 0.0
        final_answer: str | None = None
        final_retrieved_text = ""
        for rule_idx, rule in ranked:
            rules_tried.append(rule_idx)
            ok, span_text, tokens, _ = _retrieve(rule, doc)
            if not ok or tokens >= retrieval_too_large_token_threshold:
                continue
            gen = generate_answer_from_text(
                question=query_text,
                retrieved_text=span_text,
                cached_caller=cached_caller,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
            gen_calls += 1
            cum_cost += gen.cost_usd
            if _is_information_not_found(gen.answer):
                continue
            final_answer = gen.answer
            final_retrieved_text = span_text
            break

        if final_answer is None:
            rows.append(DeployedRow(
                query_idx=query_idx,
                doc_id=doc.doc_id,
                policy=_POLICY_NAME,
                rules_used=rules_tried,
                retrieved_subset_text="",
                generated_answer=None,
                judge_result=False,
                blocker="all_rules_not_found",
                gen_calls=gen_calls,
                judge_calls=0,
                actual_cost_usd=cum_cost,
                gen_cost_usd=cum_cost,
                judge_cost_usd=0.0,
                latency_ms=0.0,
            ))
            continue

        judge_pass, judge_cost = _judge_only(
            query_text,
            final_answer,
            doc.ground_truth_answer,
            cached_caller,
            llm_provider,
            llm_model,
        )
        rows.append(DeployedRow(
            query_idx=query_idx,
            doc_id=doc.doc_id,
            policy=_POLICY_NAME,
            rules_used=rules_tried,
            retrieved_subset_text=final_retrieved_text,
            generated_answer=final_answer,
            judge_result=judge_pass,
            blocker=None if judge_pass else "judge_false",
            gen_calls=gen_calls,
            judge_calls=1,
            actual_cost_usd=cum_cost + judge_cost,
            gen_cost_usd=cum_cost,
            judge_cost_usd=judge_cost,
            latency_ms=0.0,
        ))
    return rows


def _resolve_paths(config: dict[str, Any]) -> tuple[Path, Path, str | None]:
    dataset_root = config.get("dataset_root", "datasets/pdfs/latest")
    parser = config.get("parser", "docling")
    if parser == "mineru":
        return (
            Path(dataset_root) / "processing_mineru",
            Path(dataset_root) / "label_mineru",
            config.get("truncate_before"),
        )
    if parser == "docling":
        return (
            Path(dataset_root) / "processing",
            Path(dataset_root) / "label",
            config.get("truncate_before"),
        )
    raise ValueError(f"unsupported parser: {parser}")


def _sampled_doc_ids_from_payload(
    payload: dict[str, Any],
    config: dict[str, Any],
    query_idx: int,
    sampled_summary: dict[str, Any] | None = None,
    phase_a_docs: dict[str, Any] | None = None,
) -> set[str]:
    sampled_doc_ids = collect_sampled_doc_ids_from_best_rules(payload, sampled_summary)

    # Tool-agent writes exclusion IDs to a sibling phase_a_docs.json (not best_rules.json);
    # merging from there ensures cascade deploy excludes every doc the agent saw.
    if phase_a_docs is not None:
        for key in ("excluded_doc_ids", "processed_doc_ids"):
            sampled_doc_ids.update(str(d) for d in phase_a_docs.get(key) or [])

    if sampled_doc_ids:
        return sampled_doc_ids

    for qc in config.get("queries", []):
        if qc["query_idx"] == query_idx:
            return set(qc.get("documents", []))
    return set()


def _build_holdout_docs_for_query(
    query_idx: int,
    config: dict[str, Any],
    processing_dir: Path,
    label_dir: Path,
    truncate_before: str | None,
    max_holdout_docs: int,
    sampled_doc_ids: set[str],
    seed: int = 42,
) -> tuple[list[DocumentSample], list[str]]:
    label_path = label_dir / get_label_filename(
        {"dataset": config.get("dataset", "pdfs")}, query_idx
    )
    holdout_ids = select_holdout_docs(
        label_path,
        query_idx,
        processing_dir,
        sampled_doc_ids,
        max_holdout_docs,
        seed=seed,
    )
    docs = build_holdout_docs(holdout_ids, query_idx, processing_dir, label_path, truncate_before)
    return docs, holdout_ids


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Evaluate cascade deploy policy on holdout docs")
    p.add_argument("--query-idx", type=int, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--in-best-rules", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-holdout-docs", type=int, default=25)
    p.add_argument("--llm-provider", default="azure")
    p.add_argument("--llm-model", required=True)
    p.add_argument("--retrieval-too-large-token-threshold", type=int, default=5000)
    p.add_argument("--holdout-seed", type=int, default=42)
    args = p.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    payload = load_best_rules_payload(args.in_best_rules)
    sampled_summary_path = args.in_best_rules.with_name("summary.json")
    sampled_summary = None
    if sampled_summary_path.exists():
        with sampled_summary_path.open("r", encoding="utf-8") as f:
            sampled_summary = json.load(f)
    phase_a_docs_path = args.in_best_rules.with_name("phase_a_docs.json")
    phase_a_docs = None
    if phase_a_docs_path.exists():
        with phase_a_docs_path.open("r", encoding="utf-8") as f:
            phase_a_docs = json.load(f)
    sampled_doc_ids = _sampled_doc_ids_from_payload(
        payload, config, args.query_idx, sampled_summary, phase_a_docs
    )
    rules = load_rules_from_best_rules(args.in_best_rules)
    sampled_eval = load_sampled_eval_from_best_rules(args.in_best_rules, rules)
    if not rules:
        raise ValueError(f"No rules in {args.in_best_rules}")

    processing_dir, label_dir, truncate_before = _resolve_paths(config)
    holdout_docs, _ = _build_holdout_docs_for_query(
        args.query_idx,
        config,
        processing_dir,
        label_dir,
        truncate_before,
        args.max_holdout_docs,
        sampled_doc_ids,
        seed=args.holdout_seed,
    )
    print(f"[q{args.query_idx}] {len(rules)} rules, {len(holdout_docs)} holdout docs")

    query_text = get_query_text(config["dataset_root"], args.query_idx)
    cached_caller = CachedLLMCaller(DEFAULT_CACHE_DB_PATH)
    rows = evaluate_cascade(
        rules=rules,
        sampled_eval=sampled_eval,
        holdout_docs=holdout_docs,
        query_idx=args.query_idx,
        query_text=query_text,
        cached_caller=cached_caller,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        retrieval_too_large_token_threshold=args.retrieval_too_large_token_threshold,
    )
    summary = summarize_rows(rows)

    pol_dir = args.output_dir / _POLICY_NAME
    pol_dir.mkdir(parents=True, exist_ok=True)
    with (pol_dir / "deployed_rows.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (pol_dir / "deployed_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (args.output_dir / "all_policies_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"query_idx": args.query_idx, "n_holdout": len(holdout_docs), "policies": [summary]},
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"  acc={summary['deployed_acc']:.1%} cost=${summary['total_cost_usd']:.4f} "
        f"mean_gen={summary['mean_gen_calls']:.2f}"
    )


if __name__ == "__main__":
    main()
