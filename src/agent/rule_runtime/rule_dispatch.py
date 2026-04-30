"""Dispatch RangeRule | CodeRule to a uniform RetrievedSubset.

This is the only polymorphism site for Phase B rule application. The RangeRule
branch delegates directly to execute_range_rule; existing range outputs are
protected by the byte-stability hash gate.
"""

from __future__ import annotations

from typing import Union

from agent.rules.code_rule_json import CodeRule
from agent.rules.code_rule_sandbox import CodeExecResult, execute_locate_region
from agent.rules.range_rule_exec import (
    RetrievedSpan,
    RetrievedSubset,
    execute_range_rule,
)
from agent.rules.range_rule_json import RangeRule

Rule = Union[RangeRule, CodeRule]

_AST_PREFIX = "AST violations:"
_TIMEOUT_PREFIX = "TimeoutError:"


def apply_rule(rule: Rule, doc_text: str) -> RetrievedSubset:
    """Apply a RangeRule or CodeRule to document text."""
    if isinstance(rule, RangeRule):
        return execute_range_rule(rule, doc_text)
    if isinstance(rule, CodeRule):
        return _apply_code_rule(rule, doc_text)
    raise TypeError(f"unsupported rule type: {type(rule).__name__}")


def _apply_code_rule(rule: CodeRule, doc_text: str) -> RetrievedSubset:
    exec_result = execute_locate_region(rule.code, doc_text)
    region = exec_result.returned_region or ""
    matched = bool(exec_result.success) and bool(region)
    metadata: dict[str, object] = {
        "rule_kind": "code",
        "exec_time_ms": float(exec_result.exec_time_ms),
        "error": exec_result.error,
    }
    if exec_result.error_kind is not None:
        metadata["error_kind"] = exec_result.error_kind

    if matched:
        start = doc_text.find(region)
        if start < 0:
            start = 0
        span = RetrievedSpan(start=start, end=start + len(region), text=region)
        return RetrievedSubset(matched=True, spans=(span,), metadata=metadata)

    metadata["reason"] = _classify_unmatched_reason(exec_result, region)
    return RetrievedSubset(matched=False, spans=(), metadata=metadata)


def _classify_unmatched_reason(exec_result: CodeExecResult, region: str) -> str:
    if exec_result.error_kind == "ast":
        return "sandbox_rejected_ast"
    if exec_result.error_kind == "timeout":
        return "sandbox_timeout"
    err = exec_result.error or ""
    if err.startswith(_AST_PREFIX):
        return "sandbox_rejected_ast"
    if err.startswith(_TIMEOUT_PREFIX):
        return "sandbox_timeout"
    if exec_result.success and not region:
        return "sandbox_empty_region"
    if "expected non-empty str" in err:
        return "sandbox_empty_region"
    if err:
        return "sandbox_runtime_error"
    return "sandbox_runtime_error"


__all__ = ["Rule", "apply_rule"]
