"""Structured JSON parser/validator for model-generated range rules."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

# 6 retrieval modes (regex: anchor is used as a regex pattern)
ALLOWED_MODES: tuple[str, ...] = (
    "after",
    "before",
    "around",
    "between",
    "page",
    "regex",
)

RULE_FIELDS: tuple[str, ...] = (
    "rule_text",
    "evidence_basis",
    "retrieval_spec",
)

RETRIEVAL_SPEC_FIELDS: tuple[str, ...] = (
    "mode",
    "anchor",
    "anchor_b",
    "page_idx",
    "max_chars",
    "boundary_context_chars",
)
RETRIEVAL_SPEC_REQUIRED_FIELDS: tuple[str, ...] = (
    "mode",
    "anchor",
    "anchor_b",
    "page_idx",
    "max_chars",
)

_NULLABLE_STRING_SCHEMA: dict[str, Any] = {"type": ["string", "null"]}
_NULLABLE_INTEGER_SCHEMA: dict[str, Any] = {"type": ["integer", "null"]}
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


@dataclass(slots=True, frozen=True)
class RetrievalSpec:
    mode: str
    anchor: str | None
    anchor_b: str | None
    page_idx: int | None
    max_chars: int
    boundary_context_chars: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ALLOWED_MODES:
            raise ValueError(f"unknown mode: {self.mode}")
        if self.max_chars <= 0:
            raise ValueError("max_chars must be positive")
        # after / before / around require an anchor
        if self.mode in {"after", "before", "around"} and not self.anchor:
            raise ValueError(f"{self.mode} requires an anchor")
        # regex uses anchor as the pattern
        if self.mode == "regex" and not self.anchor:
            raise ValueError("regex requires an anchor (used as the regex pattern)")
        # between requires both anchors
        if self.mode == "between":
            if not self.anchor or not self.anchor_b:
                raise ValueError("between requires both anchor and anchor_b")
        # page requires page_idx
        if self.mode == "page" and self.page_idx is None:
            raise ValueError("page requires page_idx")
        if self.page_idx is not None and self.page_idx <= 0:
            raise ValueError("page_idx must be a positive integer")
        if self.boundary_context_chars < 0:
            raise ValueError("boundary_context_chars must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class RangeRule:
    rule_text: str
    evidence_basis: str
    retrieval_spec: RetrievalSpec
    # Optional regex the agent supplies during batch_apply_rule; used by holdout
    # as a 0-cost gate when phase_a_hint_reliability is high enough.
    answer_hint_pattern: str | None = None
    # Reliability of the hint pattern computed over sampled documents — reflects
    # how well the regex distinguishes judge-pass from judge-fail cases.
    # Values that are too low indicate noise; the gate is not enabled.
    phase_a_hint_reliability: float | None = None
    # Extraction precision: P(regex match.group(0) in GT | hint_match AND judge_pass).
    # A high reliability but low extraction precision means the regex fires in the
    # right region but does not isolate the answer — skip-gen would misfire.
    phase_a_hint_extraction_precision: float | None = None

    def __post_init__(self) -> None:
        if not self.rule_text.strip():
            raise ValueError("rule_text must not be empty")
        if not self.evidence_basis.strip():
            raise ValueError("evidence_basis must not be empty")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "rule_text": self.rule_text,
            "evidence_basis": self.evidence_basis,
            "retrieval_spec": self.retrieval_spec.to_dict(),
        }
        # Only persist optional fields when set; keeps files compatible with older
        # best_rules.json that lack these keys.
        if self.answer_hint_pattern is not None:
            out["answer_hint_pattern"] = self.answer_hint_pattern
        if self.phase_a_hint_reliability is not None:
            out["phase_a_hint_reliability"] = self.phase_a_hint_reliability
        if self.phase_a_hint_extraction_precision is not None:
            out["phase_a_hint_extraction_precision"] = self.phase_a_hint_extraction_precision
        return out


@dataclass(slots=True, frozen=True)
class RangeRuleBundle:
    query_idx: int
    rules: tuple[RangeRule, ...]

    def __post_init__(self) -> None:
        if self.query_idx < 0:
            raise ValueError("query_idx must be non-negative")
        if not self.rules:
            raise ValueError("rules must not be empty")
        if len(self.rules) > 5:
            raise ValueError("rules must not exceed 5 entries")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_idx": self.query_idx,
            "rules": [rule.to_dict() for rule in self.rules],
        }


def _strip_json_fence(raw_text: str) -> str:
    text = raw_text.strip()
    match = _JSON_FENCE_RE.match(text)
    if match:
        return match.group(1).strip()
    return text


def _validate_exact_fields(
    data: dict[str, Any],
    required_fields: tuple[str, ...],
    label: str,
    *,
    allowed_fields: tuple[str, ...] | None = None,
) -> None:
    data_keys = set(data.keys())
    required = set(required_fields)
    allowed = set(allowed_fields or required_fields)
    missing = required - data_keys
    extra = data_keys - allowed
    if missing:
        raise ValueError(f"{label} missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"{label} unexpected fields: {sorted(extra)}")


def _parse_retrieval_spec(data: dict[str, Any]) -> RetrievalSpec:
    _validate_exact_fields(
        data,
        RETRIEVAL_SPEC_REQUIRED_FIELDS,
        "retrieval_spec",
        allowed_fields=RETRIEVAL_SPEC_FIELDS,
    )

    mode = str(data["mode"])
    anchor = data["anchor"]
    anchor_b = data["anchor_b"]
    page_idx = data["page_idx"]
    boundary_context_chars = data.get("boundary_context_chars", 0)

    if page_idx is not None and (
        not isinstance(page_idx, int) or isinstance(page_idx, bool)
    ):
        raise ValueError("page_idx must be an integer or null")

    max_chars = data["max_chars"]
    if not isinstance(max_chars, int) or isinstance(max_chars, bool):
        raise ValueError("max_chars must be an integer")
    if not isinstance(boundary_context_chars, int) or isinstance(
        boundary_context_chars, bool
    ):
        raise ValueError("boundary_context_chars must be an integer")

    if mode in {"page", "between"} and max_chars <= 0:
        max_chars = 1

    return RetrievalSpec(
        mode=mode,
        anchor=anchor if anchor is None else str(anchor),
        anchor_b=anchor_b if anchor_b is None else str(anchor_b),
        page_idx=page_idx,
        max_chars=max_chars,
        boundary_context_chars=boundary_context_chars,
    )


def parse_range_rule_bundle(raw_text: str) -> RangeRuleBundle:
    cleaned = _strip_json_fence(raw_text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model output is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("top-level JSON must be an object")
    _validate_exact_fields(payload, ("query_idx", "rules"), "top_level")

    query_idx = payload["query_idx"]
    if not isinstance(query_idx, int) or isinstance(query_idx, bool):
        raise ValueError("query_idx must be an integer")

    rules_payload = payload["rules"]
    if not isinstance(rules_payload, list):
        raise ValueError("rules must be an array")

    rules: list[RangeRule] = []
    skipped: list[str] = []
    for idx, rule_payload in enumerate(rules_payload, start=1):
        if not isinstance(rule_payload, dict):
            raise ValueError(f"rules[{idx}] must be an object")
        _validate_exact_fields(rule_payload, RULE_FIELDS, f"rules[{idx}]")
        try:
            rules.append(
                RangeRule(
                    rule_text=str(rule_payload["rule_text"]),
                    evidence_basis=str(rule_payload["evidence_basis"]),
                    retrieval_spec=_parse_retrieval_spec(rule_payload["retrieval_spec"]),
                )
            )
        except ValueError as exc:
            skipped.append(f"rules[{idx}]: {exc}")

    if not rules:
        raise ValueError(f"all {len(rules_payload)} rules are invalid: {skipped}")

    return RangeRuleBundle(query_idx=query_idx, rules=tuple(rules))


def build_range_rule_response_schema() -> dict[str, Any]:
    """Build the structured-output JSON schema (6 modes + boundary context)."""
    return {
        "name": "range_rule_bundle",
        "description": "Executable retrieval rules for a single query over document text.",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query_idx": {"type": "integer"},
                "rules": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "rule_text": {"type": "string"},
                            "evidence_basis": {"type": "string"},
                            "retrieval_spec": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "mode": {
                                        "type": "string",
                                        "enum": list(ALLOWED_MODES),
                                    },
                                    "anchor": dict(_NULLABLE_STRING_SCHEMA),
                                    "anchor_b": dict(_NULLABLE_STRING_SCHEMA),
                                    "page_idx": dict(_NULLABLE_INTEGER_SCHEMA),
                                    "max_chars": {"type": "integer"},
                                    "boundary_context_chars": {"type": "integer"},
                                },
                                "required": list(RETRIEVAL_SPEC_FIELDS),
                            },
                        },
                        "required": list(RULE_FIELDS),
                    },
                },
            },
            "required": ["query_idx", "rules"],
        },
    }


__all__ = [
    "ALLOWED_MODES",
    "RETRIEVAL_SPEC_FIELDS",
    "RETRIEVAL_SPEC_REQUIRED_FIELDS",
    "RULE_FIELDS",
    "RangeRule",
    "RangeRuleBundle",
    "RetrievalSpec",
    "build_range_rule_response_schema",
    "parse_range_rule_bundle",
]
