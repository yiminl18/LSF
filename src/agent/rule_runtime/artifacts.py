"""Shared best_rules.json artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.rules.range_rule_json import RangeRule, RetrievalSpec


def load_best_rules_payload(best_rules_path: Path) -> dict[str, Any]:
    with best_rules_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{best_rules_path} must contain a JSON object")
    return payload


def write_best_rules_payload(best_rules_path: Path, payload: dict[str, Any]) -> None:
    best_rules_path.parent.mkdir(parents=True, exist_ok=True)
    with best_rules_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def rule_from_best_rules_entry(entry: dict[str, Any]) -> RangeRule:
    spec = RetrievalSpec(**entry["retrieval_spec"])
    return RangeRule(
        rule_text=entry.get("rule_text", "rule"),
        evidence_basis=entry.get("evidence_basis", "baseline"),
        retrieval_spec=spec,
        answer_hint_pattern=entry.get("answer_hint_pattern"),
        phase_a_hint_reliability=entry.get("phase_a_hint_reliability"),
        phase_a_hint_extraction_precision=entry.get("phase_a_hint_extraction_precision"),
    )


def collect_sampled_doc_ids_from_best_rules(
    payload: dict[str, Any],
    sampled_summary: dict[str, Any] | None = None,
) -> set[str]:
    """Collect every document that may have influenced Phase A rule generation."""
    sampled_doc_ids: set[str] = set()

    for entry in payload.get("merged_rules", []):
        sampled_doc_ids.update(str(doc_id) for doc_id in entry.get("primary_doc_ids", []))
        for doc_ids in entry.get("source_bundle_doc_ids_list", []):
            sampled_doc_ids.update(str(doc_id) for doc_id in doc_ids)

    if sampled_summary is not None:
        sampled_doc_ids.update(
            str(doc_id) for doc_id in sampled_summary.get("selected_doc_ids", [])
        )

    return sampled_doc_ids


def load_rules_from_best_rules(best_rules_path: Path) -> list[RangeRule]:
    """Construct RangeRule list from a baseline or tool-agent best_rules.json."""
    data = load_best_rules_payload(best_rules_path)
    return [rule_from_best_rules_entry(entry) for entry in data.get("merged_rules", [])]


def _cross_doc_eval_entries(data: dict[str, Any]) -> list[dict[str, Any]]:
    cross_doc_eval: list[dict[str, Any]] = []
    if isinstance(data.get("set_cover_meta"), dict):
        cross_doc_eval = data["set_cover_meta"].get("cross_doc_eval", []) or []
    if not cross_doc_eval:
        cross_doc_eval = data.get("cross_doc_eval", []) or []
    return cross_doc_eval


def _eval_stats(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": float(entry.get("accuracy", 0.0)),
        "coverage": float(entry.get("coverage", 0.0)),
        "score": float(entry.get("score", 0.0)),
        "success_doc_ids": list(entry.get("success_doc_ids", []) or []),
    }


def load_sampled_eval_from_best_rules(
    best_rules_path: Path,
    rules: list[RangeRule] | None = None,
) -> dict[int, dict[str, Any]]:
    """Read sampled cross-doc metrics from a best_rules.json artifact.

    When ``rules`` is provided, metrics are matched by retrieval_spec. This is
    required for tool-agent artifacts where top-level cross_doc_eval covers all
    unique rules, while merged_rules contains the selected deployable subset.
    """
    data = load_best_rules_payload(best_rules_path)
    cross_doc_eval = _cross_doc_eval_entries(data)

    if rules is not None:
        spec_to_eval: dict[str, dict[str, Any]] = {}
        for entry in cross_doc_eval:
            spec = entry.get("retrieval_spec")
            if spec is None:
                continue
            spec_to_eval[json.dumps(spec, sort_keys=True)] = _eval_stats(entry)
        sampled_eval: dict[int, dict[str, Any]] = {}
        for idx, rule in enumerate(rules):
            spec_key = json.dumps(rule.retrieval_spec.to_dict(), sort_keys=True)
            sampled_eval[idx] = spec_to_eval.get(
                spec_key,
                {"accuracy": 0.0, "score": 0.0, "coverage": 0.0, "success_doc_ids": []},
            )
        return sampled_eval

    sampled_eval: dict[int, dict[str, Any]] = {}
    for entry in cross_doc_eval:
        ri = entry.get("rule_index")
        if ri is None:
            continue
        sampled_eval[int(ri)] = _eval_stats(entry)
    return sampled_eval


def rank_rules_by_sampled_acc(
    rules: list[RangeRule],
    sampled_eval: dict[int, dict[str, Any]],
    key: str = "accuracy",
) -> list[tuple[int, RangeRule]]:
    """Sort rules by sampled signal desc, then rule index asc."""
    indexed = list(enumerate(rules))

    def sort_key(item: tuple[int, RangeRule]) -> tuple[float, float, int]:
        ri, _ = item
        stats = sampled_eval.get(ri, {})
        return (
            -float(stats.get(key, 0.0)),
            -float(stats.get("score", 0.0)),
            ri,
        )

    return sorted(indexed, key=sort_key)
