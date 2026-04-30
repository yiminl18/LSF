"""Parse LLM output into CodeRuleBundle — a Python locate_region function.

The LLM is expected to return a single ```python ... ``` code block containing
exactly one function named ``locate_region``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agent.rules.code_rule_sandbox import validate_code_ast

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

_LOCATE_REGION_DEF_RE = re.compile(r"^\s*def\s+locate_region\s*\(", re.MULTILINE)


@dataclass(slots=True, frozen=True)
class CodeRule:
    """A single scope-narrowing rule expressed as Python code."""

    rule_text: str
    evidence_basis: str
    code: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_text": self.rule_text,
            "evidence_basis": self.evidence_basis,
            "code": self.code,
        }


@dataclass(slots=True, frozen=True)
class CodeRuleBundle:
    """One or more CodeRules generated for a single query."""

    query_idx: int
    rules: tuple[CodeRule, ...]

    def __post_init__(self) -> None:
        if self.query_idx < 0:
            raise ValueError("query_idx must be non-negative")
        if not self.rules:
            raise ValueError("rules must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_idx": self.query_idx,
            "rules": [r.to_dict() for r in self.rules],
        }


def _extract_code_blocks(raw_text: str) -> list[str]:
    """Extract all fenced Python code blocks from *raw_text*."""
    return [m.group(1).strip() for m in _CODE_FENCE_RE.finditer(raw_text)]


def inspect_code_rule_candidates(raw_text: str) -> list[dict[str, Any]]:
    """Return validation diagnostics for fenced locate_region candidates."""
    candidates: list[dict[str, Any]] = []
    for block_index, block in enumerate(_extract_code_blocks(raw_text)):
        has_locate_region = bool(_LOCATE_REGION_DEF_RE.search(block))
        violations = validate_code_ast(block) if has_locate_region else []
        if not has_locate_region:
            status = "skipped_no_locate_region"
        elif violations:
            status = "rejected_ast"
        else:
            status = "valid"
        candidates.append(
            {
                "block_index": block_index,
                "sandbox_validation_status": status,
                "violations": violations,
                "code": block,
            }
        )
    return candidates


def parse_code_rule_bundle(
    raw_text: str,
    query_idx: int,
) -> CodeRuleBundle:
    """Parse LLM output into a :class:`CodeRuleBundle`.

    Extraction strategy:
      1. Find all ```python ... ``` fenced blocks.
      2. Keep only those that contain ``def locate_region``.
      3. AST-validate each block; skip unsafe ones.
      4. Wrap each valid block as a :class:`CodeRule`.

    Raises ``ValueError`` if no valid code block is found.
    """
    blocks = _extract_code_blocks(raw_text)
    if not blocks:
        raise ValueError("LLM output contains no fenced Python code blocks")

    rules: list[CodeRule] = []
    for block in blocks:
        if not _LOCATE_REGION_DEF_RE.search(block):
            continue

        violations = validate_code_ast(block)
        if violations:
            continue

        rules.append(
            CodeRule(
                rule_text="LLM-generated locate_region function",
                evidence_basis="anchor-based scope narrowing from training documents",
                code=block,
            )
        )

    if not rules:
        raise ValueError(
            "No valid locate_region function found in LLM output. "
            f"Extracted {len(blocks)} code block(s) but none passed validation."
        )

    return CodeRuleBundle(query_idx=query_idx, rules=tuple(rules))


__all__ = [
    "CodeRule",
    "CodeRuleBundle",
    "inspect_code_rule_candidates",
    "parse_code_rule_bundle",
]
