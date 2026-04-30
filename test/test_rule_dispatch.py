from __future__ import annotations

from dataclasses import asdict

import pytest

from agent.rule_runtime import rule_dispatch
from agent.rules.code_rule_json import CodeRule
from agent.rules.code_rule_sandbox import CodeExecResult
from agent.rules.range_rule_exec import execute_range_rule
from agent.rules.range_rule_json import RangeRule, RetrievalSpec


def _range_rule() -> RangeRule:
    return RangeRule(
        rule_text="find phone",
        evidence_basis="test",
        retrieval_spec=RetrievalSpec(
            mode="regex",
            anchor=r"\(\d{3}\) \d{3}-\d{4}",
            anchor_b=None,
            page_idx=None,
            max_chars=32,
            boundary_context_chars=0,
        ),
    )


def _code_rule(code: str = "def locate_region(document_text):\n    return document_text\n") -> CodeRule:
    return CodeRule(rule_text="code", evidence_basis="test", code=code)


def test_apply_rule_range_branch_unchanged() -> None:
    rule = _range_rule()
    text = "Registrant telephone number is (555) 123-4567."

    result = rule_dispatch.apply_rule(rule, text)
    expected = execute_range_rule(rule, text)

    assert result == expected


def test_apply_rule_code_branch_matched(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_execute(code: str, doc_text: str) -> CodeExecResult:
        return CodeExecResult(True, "needle", None, 1.25)

    monkeypatch.setattr(rule_dispatch, "execute_locate_region", fake_execute)

    result = rule_dispatch.apply_rule(_code_rule(), "prefix needle suffix")

    assert result.matched is True
    assert len(result.spans) == 1
    assert result.spans[0].start == 7
    assert result.spans[0].text == "needle"
    assert result.metadata == {
        "rule_kind": "code",
        "exec_time_ms": 1.25,
        "error": None,
    }


@pytest.mark.parametrize(
    ("exec_result", "reason"),
    [
        (
            CodeExecResult(False, "", "blocked by sandbox", 0.1, "ast"),
            "sandbox_rejected_ast",
        ),
        (
            CodeExecResult(False, "", "too slow", 5000.0, "timeout"),
            "sandbox_timeout",
        ),
        (
            CodeExecResult(False, "", "ValueError: bad", 0.1),
            "sandbox_runtime_error",
        ),
        (
            CodeExecResult(False, "", "locate_region returned str, expected non-empty str", 0.1),
            "sandbox_empty_region",
        ),
    ],
)
def test_apply_rule_code_branch_unmatched_reasons(
    monkeypatch: pytest.MonkeyPatch,
    exec_result: CodeExecResult,
    reason: str,
) -> None:
    monkeypatch.setattr(
        rule_dispatch,
        "execute_locate_region",
        lambda code, doc_text: exec_result,
    )

    result = rule_dispatch.apply_rule(_code_rule(), "text")

    assert result.matched is False
    assert result.spans == ()
    assert result.metadata["reason"] == reason


def test_apply_rule_code_branch_unmatched_reason_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rule_dispatch,
        "execute_locate_region",
        lambda code, doc_text: CodeExecResult(True, "", None, 0.1),
    )

    result = rule_dispatch.apply_rule(_code_rule(), "text")

    assert result.metadata["reason"] == "sandbox_empty_region"


def test_apply_rule_retrieved_subset_field_set() -> None:
    result = rule_dispatch.apply_rule(_range_rule(), "phone (555) 123-4567")

    assert set(asdict(result)) == {"matched", "spans", "metadata"}
