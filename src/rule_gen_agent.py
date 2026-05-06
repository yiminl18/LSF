"""Rule generation via pattern-first agent with fast substring feedback.

Improvements over rule_gen_agent_coarse_claude.py:
- Coverage-first rule design: broad rules first, targeted rules for remaining gaps
- Explicit merge accuracy target >= 0.9
- Fast feedback loop: per-rule testing uses substring match (no LLM)
- LLM calls reserved only for final test_union verification
- Diagnosis-driven refinement: structural diff between failing and passing docs
"""

from __future__ import annotations

import importlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from agent import _azure_llm, run_agent

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_AGENT_SYSTEM_PROMPT = """\
You are a document rule engineer. Generate Python span-retrieval rules for financial documents.

DOCUMENT STRUCTURE
Each document is a dict with a "texts" list of span dicts:
  text            — raw string content (tables in Markdown pipe format)
  label           — "text" | "section_header" | "table" | "list_item"
  page_no         — integer
  bold            — 1 if bold, else 0
  size            — font size in points
  structure.level — "H1" | "H2" | "H3" | "H4" | "Body"
  structure.path_text — breadcrumb of ancestor section headers, pipe-separated
  table_data.cells — list of {{row, col, text, is_column_header, is_row_header}}
                     (only present when label == "table")

RULE INTERFACE
  def rule_<name>(doc: dict) -> list[dict]:
      \"\"\"One-line description.\"\"\"
      return [span for span in doc["texts"] if ...]
Rules must be self-contained (import inside function). Never raise. Return [] on failure.

PRIMARY OBJECTIVES (in order of priority)
  1. merge_accuracy >= 0.90
     Fraction of docs where the LLM, given the UNION of text retrieved by ALL
     your rules, produces the correct answer.
  2. avg_cost_ratio as small as possible
     avg_cost_ratio = mean over all docs of:
       (tokens in union of retrieved spans) / (total tokens in document)
     A rule that returns [] for a doc contributes nothing. Rules must retrieve the
     answer-containing span, but avoid retrieving entire sections unnecessarily.

RULE WRITING STRATEGY
  - Each rule should cover as many sampled documents as possible.
  - Write rules as precisely as possible: filter by page_no, label, level,
    bold, path_text. Fewer spans returned = lower cost.
  - After test_rule(), check BOTH hit rate AND avg cost_ratio.
    If cost_ratio > 0.05, tighten filter conditions to retrieve fewer spans.
  - Never modify a rule that is already hitting its documents. Add new rules for gaps.
  - After achieving merge_accuracy >= 0.90, enter cost reduction phase:
    for each rule with avg cost_ratio > 0.05, tighten conditions and confirm
    accuracy is maintained via test_union() before keeping the change.
  - Do not define helper functions outside the rule function body.
  - Prefer path_text anchoring over keyword-only matching.

MANDATORY WORKFLOW — follow this order exactly:
1. Call summarize_answer_locations() FIRST. Do not write any rule before this.
2. Study the pattern table. Identify the dominant structural pattern (the one
   covering most docs). Write one broad rule for it. State your plan in text.
3. Call write_rule() for the broad rule.
4. Call test_rule() to check hit rate AND avg cost_ratio.
5. Call show_uncovered_docs() to see remaining gaps.
6. For each uncovered doc, call diagnose_failing_doc() to understand why.
7. Write a targeted rule for the uncovered structural variant.
8. Repeat steps 4–7 until show_uncovered_docs() returns 0.
9. Call test_union() — check merge_accuracy AND avg_cost_ratio.
10. If merge_accuracy < 0.90: fix failing docs, repeat from step 4.
11. If merge_accuracy >= 0.90 but avg_cost_ratio > 0.05:
    tighten high-cost rules (test_union flagged them), call test_union() to
    verify accuracy preserved, repeat until cost stable or no further reduction.

ADDITIONAL RULES:
- Each rule must have a unique name. Call write_rule() for all new/updated rules.
- Do not define helper functions outside the rule function body.
- Prefer path_text anchoring over keyword-only matching for structural questions."""

_QA_SYSTEM_PROMPT = """\
You are a financial document QA assistant.
Given retrieved spans from a financial filing and a question, answer precisely.
If the spans do not contain enough information, reply exactly: NOT FOUND
Return only the answer value — no explanation, no sentence."""

_JUDGE_SYSTEM_PROMPT = """\
You are an answer equivalence judge for a financial document QA system.
Judge whether the predicted answer is semantically equivalent to the ground truth.

Equivalence rules:
- Ignore minor formatting differences (dollar signs, commas, whitespace)
- "$4.5 billion" == "4,500 million" if numerically equal
- "NYSE" == "New York Stock Exchange"
- "37,684 million" == "$37,684,000,000"
- If predicted is "NOT FOUND" or empty, always judge INCORRECT

Reply with exactly one word: CORRECT or INCORRECT"""

# ---------------------------------------------------------------------------
# Helpers (reused from rule_gen_agent_coarse_claude.py)
# ---------------------------------------------------------------------------

def _make_question_slug(question: str) -> str:
    slug = question.lower()
    slug = re.sub(r"[^\w\s]", "", slug)
    slug = re.sub(r"\s+", "_", slug)
    return slug[:60]


def _extract_functions(text: str) -> list[tuple[str, str, str]]:
    """Return (func_name, description, source) triples from LLM output."""
    text = re.sub(r"```(?:python)?\n?", "", text)
    text = re.sub(r"```\n?", "", text)
    chunks = re.split(r"(?=^def rule_)", text, flags=re.MULTILINE)
    results: list[tuple[str, str, str]] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk.startswith("def rule_"):
            continue
        m = re.match(r"def\s+(rule_\w+)\s*\(", chunk)
        if not m:
            continue
        func_name = m.group(1)
        doc_m = re.search(r'"""(.*?)"""', chunk, re.DOTALL)
        if not doc_m:
            doc_m = re.search(r"'''(.*?)'''", chunk, re.DOTALL)
        description = doc_m.group(1).strip() if doc_m else ""
        results.append((func_name, description, chunk))
    return results


def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


def _sort_spans(spans: list[dict], doc: dict) -> list[dict]:
    texts = doc.get("texts", [])
    positions = {id(s): i for i, s in enumerate(texts)}

    def _key(s: dict) -> tuple:
        page = s.get("page_no", 0)
        pos = positions.get(id(s), 9999)
        return (page, pos)

    return sorted(spans, key=_key)


def _call_qa(retrieved_text: str, question: str, model_mod: Any) -> str:
    if not retrieved_text.strip():
        return "NOT FOUND"
    try:
        resp = model_mod.client.chat.completions.create(
            model=model_mod.AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _QA_SYSTEM_PROMPT},
                {"role": "user", "content": f"Passage:\n{retrieved_text}\n\nQuestion: {question}"},
            ],
            max_completion_tokens=200,
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip() or "NOT FOUND"
    except Exception as exc:
        return f"QA_ERROR: {exc}"


def _call_judge(question: str, predicted: str, ground_truth: str, model_mod: Any) -> bool:
    if not predicted or predicted in ("NOT FOUND", "") or predicted.startswith("QA_ERROR"):
        return False
    try:
        resp = model_mod.client.chat.completions.create(
            model=model_mod.AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Question: {question}\n"
                    f"Predicted: {predicted}\n"
                    f"Ground truth: {ground_truth}"
                )},
            ],
            max_completion_tokens=10,
            temperature=0.0,
        )
        verdict = (resp.choices[0].message.content or "").strip().upper()
        return verdict.startswith("CORRECT")
    except Exception:
        return False


def _call_qa_tracked(
    retrieved_text: str, question: str, model_mod: Any
) -> tuple[str, int, int]:
    """Like _call_qa but returns (answer, input_tokens, output_tokens)."""
    if not retrieved_text.strip():
        return "NOT FOUND", 0, 0
    try:
        resp = model_mod.client.chat.completions.create(
            model=model_mod.AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _QA_SYSTEM_PROMPT},
                {"role": "user", "content": f"Passage:\n{retrieved_text}\n\nQuestion: {question}"},
            ],
            max_completion_tokens=200,
            temperature=0.0,
        )
        ans = (resp.choices[0].message.content or "").strip() or "NOT FOUND"
        in_tok = getattr(resp.usage, "prompt_tokens", 0) or 0
        out_tok = getattr(resp.usage, "completion_tokens", 0) or 0
        return ans, in_tok, out_tok
    except Exception as exc:
        return f"QA_ERROR: {exc}", 0, 0


def _call_judge_tracked(
    question: str, predicted: str, ground_truth: str, model_mod: Any
) -> tuple[bool, int, int]:
    """Like _call_judge but returns (correct, input_tokens, output_tokens)."""
    if not predicted or predicted in ("NOT FOUND", "") or predicted.startswith("QA_ERROR"):
        return False, 0, 0
    try:
        resp = model_mod.client.chat.completions.create(
            model=model_mod.AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Question: {question}\n"
                    f"Predicted: {predicted}\n"
                    f"Ground truth: {ground_truth}"
                )},
            ],
            max_completion_tokens=10,
            temperature=0.0,
        )
        verdict = (resp.choices[0].message.content or "").strip().upper()
        in_tok = getattr(resp.usage, "prompt_tokens", 0) or 0
        out_tok = getattr(resp.usage, "completion_tokens", 0) or 0
        return verdict.startswith("CORRECT"), in_tok, out_tok
    except Exception:
        return False, 0, 0


def _run_rules_on_docs(
    rule_fns: dict[str, Any],
    documents: list[dict],
    question: str,
    ground_truth: dict,
    model_mod: Any,
    *,
    union_mode: bool = False,
    judge_tokens: dict | None = None,
) -> dict:
    """Evaluate rule functions. If union_mode=True, evaluate the union of ALL rules."""
    if union_mode:
        report: dict = {}
        num_correct = 0
        total_spans = 0
        per_doc: dict = {}
        cost_ratios: list[float] = []

        for doc in documents:
            doc_name = doc.get("doc_name", "unknown")
            filename = doc.get("origin", {}).get("filename", doc_name + ".pdf")
            gt_val = ground_truth.get(filename, ground_truth.get(doc_name))

            texts = doc.get("texts", [])
            text_positions = {id(s): i for i, s in enumerate(texts)}

            all_spans: list[dict] = []
            for fn in rule_fns.values():
                try:
                    s = fn(doc)
                    if isinstance(s, list):
                        all_spans.extend(s)
                except Exception:
                    pass

            seen: set[int] = set()
            deduped: list[dict] = []
            for span in all_spans:
                idx = text_positions.get(id(span))
                if idx is None:
                    try:
                        idx = texts.index(span)
                    except ValueError:
                        idx = None
                key = idx if idx is not None else id(span)
                if key not in seen:
                    seen.add(key)
                    deduped.append(span)

            sorted_spans = _sort_spans(deduped, doc)
            retrieved = "\n\n".join(s["text"] for s in sorted_spans)
            total_spans += len(sorted_spans)

            doc_tokens = _count_tokens("\n".join(s.get("text", "") for s in texts))
            ret_tokens = _count_tokens(retrieved)
            cost_ratio = round(ret_tokens / doc_tokens, 5) if doc_tokens > 0 else 0.0
            cost_ratios.append(cost_ratio)

            predicted, in_qa, out_qa = _call_qa_tracked(retrieved, question, model_mod)
            if judge_tokens is not None:
                judge_tokens["input"] += in_qa
                judge_tokens["output"] += out_qa
            correct, in_j, out_j = (
                _call_judge_tracked(question, predicted, str(gt_val), model_mod)
                if gt_val is not None else (False, 0, 0)
            )
            if judge_tokens is not None:
                judge_tokens["input"] += in_j
                judge_tokens["output"] += out_j
            if correct:
                num_correct += 1

            per_doc[doc_name] = {
                "predicted": predicted,
                "ground_truth": str(gt_val),
                "correct": correct,
                "num_spans": len(sorted_spans),
                "cost_ratio": cost_ratio,
                "retrieved_preview": retrieved[:400] if not correct else "(correct — omitted)",
            }

        n = len(documents)
        report["_merge"] = {
            "accuracy": round(num_correct / n, 4) if n > 0 else 0.0,
            "num_correct": num_correct,
            "num_docs": n,
            "avg_spans": round(total_spans / n, 2) if n > 0 else 0.0,
            "avg_cost_ratio": round(sum(cost_ratios) / len(cost_ratios), 5) if cost_ratios else 0.0,
            "per_doc": per_doc,
        }
        return report

    else:
        report = {}
        for rule_name, rule_fn in rule_fns.items():
            num_correct = 0
            total_spans = 0
            per_doc = {}
            cost_ratios: list[float] = []

            for doc in documents:
                doc_name = doc.get("doc_name", "unknown")
                filename = doc.get("origin", {}).get("filename", doc_name + ".pdf")
                gt_val = ground_truth.get(filename, ground_truth.get(doc_name))

                try:
                    spans = rule_fn(doc)
                    if not isinstance(spans, list):
                        spans = []
                except Exception as exc:
                    per_doc[doc_name] = {
                        "error": str(exc), "correct": False,
                        "predicted": "RULE_ERROR", "ground_truth": str(gt_val),
                    }
                    continue

                sorted_spans = _sort_spans(spans, doc)
                retrieved = "\n\n".join(s["text"] for s in sorted_spans)
                total_spans += len(sorted_spans)

                texts = doc.get("texts", [])
                doc_tokens = _count_tokens("\n".join(s.get("text", "") for s in texts))
                ret_tokens = _count_tokens(retrieved)
                cost_ratio = round(ret_tokens / doc_tokens, 5) if doc_tokens > 0 else 0.0
                cost_ratios.append(cost_ratio)

                predicted = _call_qa(retrieved, question, model_mod)
                correct = _call_judge(question, predicted, str(gt_val), model_mod) if gt_val is not None else False
                if correct:
                    num_correct += 1

                per_doc[doc_name] = {
                    "predicted": predicted,
                    "ground_truth": str(gt_val),
                    "correct": correct,
                    "num_spans": len(sorted_spans),
                    "cost_ratio": cost_ratio,
                    "retrieved_preview": retrieved[:400] if not correct else "(correct — omitted)",
                }

            n = len(documents)
            report[rule_name] = {
                "accuracy": round(num_correct / n, 4) if n > 0 else 0.0,
                "num_correct": num_correct,
                "num_docs": n,
                "avg_spans": round(total_spans / n, 2) if n > 0 else 0.0,
                "avg_cost_ratio": round(sum(cost_ratios) / len(cost_ratios), 5) if cost_ratios else 0.0,
                "per_doc": per_doc,
            }
        return report


# ---------------------------------------------------------------------------
# Shared helpers for tool internals
# ---------------------------------------------------------------------------

def _find_answer_span(doc: dict, gt_val: str) -> tuple[dict | None, int | None]:
    """Return (span, idx) of the first span containing gt_val by substring, or (None, None)."""
    gt_lower = gt_val.lower().strip()
    if not gt_lower:
        return None, None
    for i, span in enumerate(doc.get("texts", [])):
        text = (span.get("text") or "").lower().strip()
        if gt_lower in text:
            return span, i
    return None, None


def _compute_hit(doc: dict, gt_val: str, rule_fns: dict[str, Any]) -> bool:
    """True if any rule retrieves text containing gt_val (substring, case-insensitive)."""
    gt_lower = gt_val.lower().strip()
    if not gt_lower:
        return False
    for fn in rule_fns.values():
        try:
            spans = fn(doc)
            if not isinstance(spans, list):
                continue
            retrieved = "\n".join(s.get("text", "") for s in spans if isinstance(s, dict))
            if gt_lower in retrieved.lower():
                return True
        except Exception:
            pass
    return False


def _compile_registered(registered: dict[str, tuple[str, str]]) -> dict[str, Any]:
    """Compile all registered rules into callable functions."""
    fns: dict[str, Any] = {}
    for rule_name, (source, _) in registered.items():
        ns: dict = {}
        try:
            exec(compile(source, f"<{rule_name}>", "exec"), ns)
            fn = ns.get(rule_name)
            if callable(fn):
                fns[rule_name] = fn
        except Exception:
            pass
    return fns


def _inspect_answer_context_impl(
    doc_name: str,
    docs_by_name: dict[str, dict],
    ground_truth: dict,
) -> str:
    doc = docs_by_name.get(doc_name)
    if not doc:
        return f"Unknown doc_name: {doc_name!r}"
    filename = doc.get("origin", {}).get("filename", doc_name + ".pdf")
    gt_val = str(ground_truth.get(filename, ground_truth.get(doc_name, "")))
    _, answer_idx = _find_answer_span(doc, gt_val)

    if answer_idx is None:
        return f"Answer not found by substring match in {doc_name}. GT: {gt_val!r}"

    texts = doc.get("texts", [])
    start = max(0, answer_idx - 5)
    end = min(len(texts), answer_idx + 6)

    lines = []
    for i in range(start, end):
        span = texts[i]
        struct = span.get("structure") or {}
        text_preview = (span.get("text") or "")[:80]
        marker = "  ← ANSWER HERE" if i == answer_idx else ""
        lines.append(
            f"[idx={i}] text={text_preview!r}  label={span.get('label')}  "
            f"page={span.get('page_no')}  bold={span.get('bold', 0)}  "
            f"level={struct.get('level', '')}  path_text={str(struct.get('path_text', ''))!r}"
            f"{marker}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool factories
# ---------------------------------------------------------------------------

def _make_summarize_answer_locations_tool(
    documents: list[dict],
    ground_truth: dict,
) -> Any:
    @tool
    def summarize_answer_locations() -> str:
        """Compute a cross-document pattern table. For each document, locate the ground truth answer span by substring match and report its structural fields: page, label, level, bold, all_cap, path_text, answer_snippet. Rows marked NOT_LOCATED mean substring match failed — use inspect_answer_context for those docs. Call this FIRST before writing any rule."""
        rows = []
        for doc in documents:
            doc_name = doc.get("doc_name", "unknown")
            filename = doc.get("origin", {}).get("filename", doc_name + ".pdf")
            gt_val = str(ground_truth.get(filename, ground_truth.get(doc_name, "")))
            span, _ = _find_answer_span(doc, gt_val)
            if span is not None:
                struct = span.get("structure") or {}
                raw_text = span.get("text") or ""
                rows.append({
                    "doc_name": doc_name,
                    "page": span.get("page_no", ""),
                    "label": span.get("label", ""),
                    "level": struct.get("level", ""),
                    "bold": span.get("bold", 0),
                    "all_cap": int(raw_text.isupper()),
                    "path_text": (struct.get("path_text") or "")[:22],
                    "answer_snippet": raw_text[:40],
                })
            else:
                rows.append({
                    "doc_name": doc_name,
                    "page": "N/A",
                    "label": "NOT_LOCATED",
                    "level": "N/A",
                    "bold": "N/A",
                    "all_cap": "N/A",
                    "path_text": "",
                    "answer_snippet": f"(GT: {gt_val[:30]})",
                })

        header = (
            f"{'doc_name':<22} {'page':<5} {'label':<16} {'level':<7} "
            f"{'bold':<5} {'all_cap':<8} {'path_text':<23} answer_snippet"
        )
        sep = "-" * len(header)
        lines = [header, sep]
        for r in rows:
            lines.append(
                f"{r['doc_name']:<22} {str(r['page']):<5} {str(r['label']):<16} "
                f"{str(r['level']):<7} {str(r['bold']):<5} {str(r['all_cap']):<8} "
                f"{str(r['path_text']):<23} {str(r['answer_snippet'])}"
            )
        return "\n".join(lines)

    return summarize_answer_locations


def _make_inspect_answer_context_tool(
    documents: list[dict],
    ground_truth: dict,
) -> Any:
    docs_by_name: dict[str, dict] = {}
    for d in documents:
        name = d.get("doc_name", "unknown")
        filename = d.get("origin", {}).get("filename", name + ".pdf")
        docs_by_name[name] = d
        docs_by_name[filename] = d

    @tool
    def inspect_answer_context(doc_name: str) -> str:
        """Show the 5 spans before and after the answer span in the given document, with full field annotations. The answer span is marked with 'ANSWER HERE'. Useful for understanding the local structure around the answer without dumping the whole document.

        Args:
            doc_name: Document name (e.g. 'AMCOR_2019_10K').
        """
        return _inspect_answer_context_impl(doc_name, docs_by_name, ground_truth)

    return inspect_answer_context


def _make_write_rule_tool(
    registered_rules: dict[str, tuple[str, str]],
) -> Any:
    @tool
    def write_rule(rule_name: str, code: str) -> str:
        """Validate and register a candidate rule. The code must define a function named rule_name. Checks: (1) name starts with rule_, (2) code executes without error, (3) function returns list when called on empty doc, (4) no top-level imports outside function body. Returns 'OK: rule registered' or a specific rejection reason.

        Args:
            rule_name: Name of the rule function (must start with 'rule_').
            code: Python source code defining the rule function.
        """
        if not rule_name.startswith("rule_"):
            return f"Rejected: rule_name must start with 'rule_', got {rule_name!r}"

        # Check for top-level imports outside function body
        code_lines = code.strip().splitlines()
        in_func = False
        for line in code_lines:
            stripped = line.strip()
            if stripped.startswith("def " + rule_name):
                in_func = True
            if not in_func and (stripped.startswith("import ") or stripped.startswith("from ")):
                return (
                    f"Rejected: top-level import found: {stripped!r}. "
                    "Put all imports inside the function body."
                )

        # Execute in empty namespace
        ns: dict = {}
        try:
            exec(compile(code, f"<{rule_name}>", "exec"), ns)
        except Exception as exc:
            return f"Rejected: code execution error: {exc}"

        fn = ns.get(rule_name)
        if fn is None:
            return f"Rejected: function {rule_name!r} not found in submitted code"
        if not callable(fn):
            return f"Rejected: {rule_name} is not callable"

        # Test on minimal stub
        try:
            result = fn({"texts": []})
            if not isinstance(result, list):
                return (
                    f"Rejected: {rule_name}({{'texts': []}}) returned "
                    f"{type(result).__name__}, expected list"
                )
        except NameError as exc:
            return f"Rejected: NameError when testing (likely hallucinated helper): {exc}"
        except Exception as exc:
            return f"Rejected: error when testing on stub doc: {exc}"

        # Extract description from docstring
        doc_m = re.search(r'"""(.*?)"""', code, re.DOTALL)
        if not doc_m:
            doc_m = re.search(r"'''(.*?)'''", code, re.DOTALL)
        description = doc_m.group(1).strip() if doc_m else ""

        registered_rules[rule_name] = (code, description)
        return f"OK: rule registered — {rule_name}"

    return write_rule


def _make_test_rule_tool(
    documents: list[dict],
    ground_truth: dict,
    registered_rules: dict[str, tuple[str, str]],
) -> Any:
    @tool
    def test_rule(rule_name: str) -> str:
        """Test a single registered rule against all documents using substring match only — no LLM call. For each document, applies the rule and checks if the ground truth answer appears (case-insensitive) in the concatenated retrieved text. Returns per-doc hit table and aggregate hit rate.

        Args:
            rule_name: Name of a previously registered rule (via write_rule).
        """
        entry = registered_rules.get(rule_name)
        if entry is None:
            registered_names = sorted(registered_rules.keys())
            return (
                f"Unknown rule: {rule_name!r}. "
                f"Registered rules: {registered_names}. Call write_rule first."
            )

        source, _ = entry
        ns: dict = {}
        try:
            exec(compile(source, f"<{rule_name}>", "exec"), ns)
            fn = ns[rule_name]
        except Exception as exc:
            return f"Error compiling {rule_name}: {exc}"

        n = len(documents)
        hits = 0
        cost_ratios: list[float] = []
        header = f"{'doc_name':<22} {'hit':<6} {'ret_tokens':<11} {'cost_ratio':<11} snippet"
        sep = "-" * len(header)
        lines = [header, sep]

        for doc in documents:
            doc_name = doc.get("doc_name", "unknown")
            filename = doc.get("origin", {}).get("filename", doc_name + ".pdf")
            gt_val = str(ground_truth.get(filename, ground_truth.get(doc_name, "")))
            gt_lower = gt_val.lower().strip()

            try:
                spans = fn(doc)
                if not isinstance(spans, list):
                    spans = []
            except Exception:
                spans = []

            retrieved = "\n".join(s.get("text", "") for s in spans if isinstance(s, dict))
            ret_tokens = _count_tokens(retrieved)
            total_doc_tokens = _count_tokens(
                "\n".join(s.get("text", "") for s in doc.get("texts", []) if isinstance(s, dict))
            )
            cost_ratio = ret_tokens / total_doc_tokens if total_doc_tokens > 0 else 0.0
            cost_ratios.append(cost_ratio)
            hit = bool(gt_lower and gt_lower in retrieved.lower())
            if hit:
                hits += 1

            mark = "✓" if hit else "✗"
            snippet = retrieved[:100] if retrieved.strip() else "(empty)"
            snippet_repr = repr(snippet)[:60]
            lines.append(
                f"{doc_name:<22} {mark:<6} {ret_tokens:<11} {cost_ratio:<11.4f} {snippet_repr}"
            )

        lines.append("")
        avg_cost = sum(cost_ratios) / len(cost_ratios) if cost_ratios else 0.0
        lines.append(f"Hit rate: {hits}/{n} ({hits/n:.2f})   Avg cost ratio: {avg_cost:.4f}")
        return "\n".join(lines)

    return test_rule


def _make_show_uncovered_docs_tool(
    documents: list[dict],
    ground_truth: dict,
    registered_rules: dict[str, tuple[str, str]],
) -> Any:
    @tool
    def show_uncovered_docs() -> str:
        """Return which documents are NOT covered by any currently registered rule (substring match, no LLM). A document is covered if at least one registered rule retrieves text containing the ground truth answer."""
        compiled = _compile_registered(registered_rules)
        n = len(documents)
        all_names = [d.get("doc_name", "unknown") for d in documents]
        covered: set[str] = set()

        for doc in documents:
            doc_name = doc.get("doc_name", "unknown")
            filename = doc.get("origin", {}).get("filename", doc_name + ".pdf")
            gt_val = str(ground_truth.get(filename, ground_truth.get(doc_name, "")))
            if _compute_hit(doc, gt_val, compiled):
                covered.add(doc_name)

        uncovered = [name for name in all_names if name not in covered]
        if not uncovered:
            return f"All {n} documents covered. Covered: {n}/{n}"

        lines = ["Uncovered (no registered rule hits):"]
        for name in uncovered:
            lines.append(f"  - {name}")
        lines.append(f"\nCovered: {n - len(uncovered)}/{n}")
        return "\n".join(lines)

    return show_uncovered_docs


def _make_diagnose_failing_doc_tool(
    documents: list[dict],
    ground_truth: dict,
    registered_rules: dict[str, tuple[str, str]],
) -> Any:
    docs_by_name: dict[str, dict] = {}
    for d in documents:
        name = d.get("doc_name", "unknown")
        filename = d.get("origin", {}).get("filename", name + ".pdf")
        docs_by_name[name] = d
        docs_by_name[filename] = d

    @tool
    def diagnose_failing_doc(doc_name: str) -> str:
        """For a document that is uncovered, show: (1) what each registered rule retrieved from this document, (2) the answer context (spans around the answer), (3) a structural diff comparing this document's answer span against documents where rules succeed.

        Args:
            doc_name: Name of an uncovered document.
        """
        doc = docs_by_name.get(doc_name)
        if not doc:
            available = [d.get("doc_name", "?") for d in documents]
            return f"Unknown doc_name: {doc_name!r}. Available: {available}"

        filename = doc.get("origin", {}).get("filename", doc_name + ".pdf")
        gt_val = str(ground_truth.get(filename, ground_truth.get(doc_name, "")))

        lines = [f"=== Diagnosis: {doc_name} ===", "", f"Ground truth: {gt_val!r}", ""]

        # 1. Rule results on this doc
        lines.append("Rule results:")
        if not registered_rules:
            lines.append("  (no rules registered yet)")
        else:
            for rule_name, (source, _) in registered_rules.items():
                ns: dict = {}
                try:
                    exec(compile(source, f"<{rule_name}>", "exec"), ns)
                    fn = ns[rule_name]
                    spans = fn(doc)
                    if isinstance(spans, list) and spans:
                        retrieved = "\n".join(
                            s.get("text", "") for s in spans if isinstance(s, dict)
                        )
                        snippet = repr(retrieved[:100])
                        lines.append(f"  {rule_name}:  retrieved={snippet}")
                    else:
                        lines.append(f"  {rule_name}:  retrieved=[]  (empty)")
                except Exception as exc:
                    lines.append(f"  {rule_name}:  ERROR: {exc}")
        lines.append("")

        # 2. Answer context
        lines.append("Answer location (inspect_answer_context):")
        ctx = _inspect_answer_context_impl(doc_name, docs_by_name, ground_truth)
        for ln in ctx.splitlines():
            lines.append("  " + ln)
        lines.append("")

        # 3. Structural diff vs passing docs
        compiled = _compile_registered(registered_rules)
        passing_docs: list[dict] = []
        for d in documents:
            dname = d.get("doc_name", "unknown")
            if dname == doc_name:
                continue
            fname = d.get("origin", {}).get("filename", dname + ".pdf")
            dgt = str(ground_truth.get(fname, ground_truth.get(dname, "")))
            if _compute_hit(d, dgt, compiled):
                passing_docs.append(d)

        if passing_docs and compiled:
            lines.append(f"Structural diff vs. passing docs ({len(passing_docs)} passing):")
            fail_span, _ = _find_answer_span(doc, gt_val)
            if fail_span:
                struct_f = fail_span.get("structure") or {}
                raw_f = fail_span.get("text") or ""
                lines.append(f"  This doc ({doc_name}):")
                lines.append(
                    f"    label={fail_span.get('label')}  level={struct_f.get('level', '')}  "
                    f"bold={fail_span.get('bold', 0)}  all_cap={int(raw_f.isupper())}  "
                    f"path_text={str(struct_f.get('path_text', ''))!r}"
                )
            else:
                lines.append(f"  This doc ({doc_name}): answer span NOT LOCATED by substring match")

            lines.append(f"  Passing docs (up to 3 shown):")
            for pd in passing_docs[:3]:
                pdname = pd.get("doc_name", "?")
                pfname = pd.get("origin", {}).get("filename", pdname + ".pdf")
                pgt = str(ground_truth.get(pfname, ground_truth.get(pdname, "")))
                pspan, _ = _find_answer_span(pd, pgt)
                if pspan:
                    struct_p = pspan.get("structure") or {}
                    raw_p = pspan.get("text") or ""
                    lines.append(
                        f"    {pdname}: label={pspan.get('label')}  "
                        f"level={struct_p.get('level', '')}  "
                        f"bold={pspan.get('bold', 0)}  all_cap={int(raw_p.isupper())}  "
                        f"path_text={str(struct_p.get('path_text', ''))!r}"
                    )
                else:
                    lines.append(f"    {pdname}: answer span not locatable by substring")
        elif not compiled:
            lines.append("No registered rules to compare (register at least one rule first).")
        else:
            lines.append("No passing docs found yet — no rule covers any document.")

        return "\n".join(lines)

    return diagnose_failing_doc


def _make_test_union_tool(
    documents: list[dict],
    question: str,
    ground_truth: dict,
    model_mod: Any,
    registered_rules: dict[str, tuple[str, str]],
    total_llm_calls: list[int],
    judge_tokens: dict,
) -> Any:
    @tool
    def test_union() -> str:
        """Test the union of ALL currently registered rules using LLM QA + LLM judge — the true merge accuracy metric. Call sparingly (only when coverage looks complete or for a checkpoint). For each document, unions spans from all rules, calls LLM QA then LLM judge. Returns merge accuracy, avg cost ratio, per-doc cost, and high-cost rule warnings."""
        if not registered_rules:
            return "No registered rules. Call write_rule first."

        compiled = _compile_registered(registered_rules)
        if not compiled:
            return "No valid rule functions could be compiled. Check write_rule output."

        n = len(documents)
        num_correct = 0
        per_doc_results: list[dict] = []
        union_cost_ratios: list[float] = []

        for doc in documents:
            doc_name = doc.get("doc_name", "unknown")
            filename = doc.get("origin", {}).get("filename", doc_name + ".pdf")
            gt_val = ground_truth.get(filename, ground_truth.get(doc_name))

            texts = doc.get("texts", [])
            text_positions = {id(s): i for i, s in enumerate(texts)}

            all_spans: list[dict] = []
            for fn in compiled.values():
                try:
                    s = fn(doc)
                    if isinstance(s, list):
                        all_spans.extend(s)
                except Exception:
                    pass

            seen: set[int] = set()
            deduped: list[dict] = []
            for span in all_spans:
                idx = text_positions.get(id(span))
                if idx is None:
                    try:
                        idx = texts.index(span)
                    except ValueError:
                        idx = None
                key = idx if idx is not None else id(span)
                if key not in seen:
                    seen.add(key)
                    deduped.append(span)

            sorted_spans = _sort_spans(deduped, doc)
            retrieved = "\n\n".join(s["text"] for s in sorted_spans)

            doc_tokens = _count_tokens("\n".join(s.get("text", "") for s in texts))
            ret_tokens = _count_tokens(retrieved)
            cost_ratio = round(ret_tokens / doc_tokens, 5) if doc_tokens > 0 else 0.0
            union_cost_ratios.append(cost_ratio)

            total_llm_calls[0] += 1
            predicted, in_qa, out_qa = _call_qa_tracked(retrieved, question, model_mod)
            judge_tokens["input"] += in_qa
            judge_tokens["output"] += out_qa
            total_llm_calls[0] += 1
            correct, in_j, out_j = (
                _call_judge_tracked(question, predicted, str(gt_val), model_mod)
                if gt_val is not None else (False, 0, 0)
            )
            judge_tokens["input"] += in_j
            judge_tokens["output"] += out_j
            if correct:
                num_correct += 1

            per_doc_results.append({
                "doc_name": doc_name,
                "correct": correct,
                "cost_ratio": cost_ratio,
                "predicted": predicted,
                "ground_truth": str(gt_val),
                "retrieved_preview": retrieved[:200] if not correct else "(correct)",
            })

        accuracy = round(num_correct / n, 4) if n > 0 else 0.0
        avg_union_cost = round(sum(union_cost_ratios) / len(union_cost_ratios), 5) if union_cost_ratios else 0.0
        lines = [f"Merge accuracy: {num_correct}/{n} ({accuracy})  Avg cost ratio: {avg_union_cost:.5f}"]

        failing = [r for r in per_doc_results if not r["correct"]]
        if not failing:
            lines.append("All documents correct!")
        else:
            lines.append("")
            lines.append("Failing docs:")
            for r in failing:
                lines.append(f"  {r['doc_name']}:")
                lines.append(f"    retrieved: {r['retrieved_preview']!r:.120}")
                lines.append(f"    predicted: {r['predicted']!r}")
                lines.append(f"    ground_truth: {r['ground_truth']!r}")

        lines.append("")
        lines.append("Per-doc cost:")
        for r in per_doc_results:
            status = "✓" if r["correct"] else "✗"
            lines.append(f"  {status} {r['doc_name']:<40} cost_ratio={r['cost_ratio']:.5f}")

        # Per-rule cost (token counting only, no LLM)
        high_cost: list[tuple[str, float]] = []
        for rule_name, fn in compiled.items():
            rule_costs: list[float] = []
            for doc in documents:
                texts = doc.get("texts", [])
                doc_tokens = _count_tokens("\n".join(s.get("text", "") for s in texts))
                try:
                    spans = fn(doc)
                    if not isinstance(spans, list):
                        spans = []
                except Exception:
                    spans = []
                retrieved = "\n".join(s.get("text", "") for s in spans if isinstance(s, dict))
                ret_tokens = _count_tokens(retrieved)
                rule_costs.append(ret_tokens / doc_tokens if doc_tokens > 0 else 0.0)
            avg_rc = sum(rule_costs) / len(rule_costs) if rule_costs else 0.0
            if avg_rc > 0.05:
                high_cost.append((rule_name, avg_rc))

        if high_cost:
            lines.append("")
            lines.append("High-cost rules (avg cost_ratio > 0.05) — consider tightening:")
            for rname, rc in sorted(high_cost, key=lambda x: -x[1]):
                lines.append(f"  {rname:<50}  avg_cost={rc:.4f}")

        return "\n".join(lines)

    return test_union


# ---------------------------------------------------------------------------
# User prompt
# ---------------------------------------------------------------------------

def _build_user_prompt(documents: list[dict], question: str, ground_truth: dict) -> str:
    n = len(documents)
    target_stmt = (
        f"TARGET:\n"
        f"  1. merge_accuracy >= 0.90  (primary)\n"
        f"     Fraction of docs correctly answered using the UNION of all rule outputs.\n"
        f"  2. avg_cost_ratio as small as possible  (secondary)\n"
        f"     avg_cost_ratio = mean over docs of:\n"
        f"       (tokens in union of retrieved spans) / (total tokens in document)\n"
        f"Each rule should cover as many documents as possible. Write rules as precisely\n"
        f"as possible: filter by page_no, label, level, bold, path_text.\n"
        f"After achieving merge_accuracy >= 0.90, tighten high-cost rules while\n"
        f"verifying accuracy via test_union().\n\n"
    )

    parts: list[str] = [
        target_stmt,
        f"QUESTION: {question}\n\n",
        "Ground truth answers:\n",
    ]
    for doc in documents:
        doc_name = doc.get("doc_name", "unknown")
        filename = doc.get("origin", {}).get("filename", doc_name + ".pdf")
        answer = ground_truth.get(filename, ground_truth.get(doc_name, "N/A"))
        parts.append(f"  {filename}: {answer}\n")

    parts.append(
        "\nFor each rule, consider the following signal types as hints:\n"
        "1. PHYSICAL LOCATION — Which page(s) does the answer consistently appear on?\n"
        "2. SEMANTIC LOCATION — Which section header is the answer under? Use structure.path_text.\n"
        "3. KEYWORD PROXIMITY — What keywords appear near the answer?\n"
        "4. DATA FEATURE — Is the answer in a table? Use label=='table' and table_data.cells.\n"
        "5. TYPOGRAPHY — Is the answer in a bold span, large font, or all-caps heading?\n"
        "6. STRUCTURAL POSITION — What is the heading level or depth of the span?\n"
        "7. ANY OTHER pattern you observe — label combinations, sibling relationships, etc.\n\n"
    )

    parts.append("Document spans (first 80 per document):\n")
    for doc in documents:
        doc_name = doc.get("doc_name", "unknown")
        filename = doc.get("origin", {}).get("filename", doc_name + ".pdf")
        answer = ground_truth.get(filename, ground_truth.get(doc_name, "N/A"))
        spans_json = json.dumps(doc["texts"][:80], indent=2)
        parts.append(
            f"\n--- Document: {doc_name} ---\n"
            f"Answer: {answer}\n"
            f"Spans:\n{spans_json}\n"
        )

    parts.append(
        "\nStart by calling summarize_answer_locations() to see the pattern table.\n"
        "Then follow the mandatory workflow in the system prompt."
    )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Post-run: compute per-rule stats
# ---------------------------------------------------------------------------

def _compute_rule_stats(
    registered_rules: dict[str, tuple[str, str]],
    documents: list[dict],
    ground_truth: dict,
) -> dict[str, dict]:
    """Return {rule_name: {"hit_rate": float, "avg_cost_ratio": float}} using substring match."""
    n = len(documents)
    if n == 0:
        return {}

    stats: dict[str, dict] = {}
    for rule_name, (source, _) in registered_rules.items():
        ns: dict = {}
        try:
            exec(compile(source, f"<{rule_name}>", "exec"), ns)
            fn = ns.get(rule_name)
        except Exception:
            stats[rule_name] = {"hit_rate": 0.0, "avg_cost_ratio": 0.0}
            continue
        if not callable(fn):
            stats[rule_name] = {"hit_rate": 0.0, "avg_cost_ratio": 0.0}
            continue

        hits = 0
        cost_ratios: list[float] = []
        for doc in documents:
            doc_name = doc.get("doc_name", "unknown")
            filename = doc.get("origin", {}).get("filename", doc_name + ".pdf")
            gt_val = str(ground_truth.get(filename, ground_truth.get(doc_name, "")))
            gt_lower = gt_val.lower().strip()
            try:
                spans = fn(doc)
                if not isinstance(spans, list):
                    spans = []
                retrieved = "\n".join(s.get("text", "") for s in spans if isinstance(s, dict))
                texts = doc.get("texts", [])
                doc_tokens = _count_tokens("\n".join(s.get("text", "") for s in texts))
                ret_tokens = _count_tokens(retrieved)
                cost_ratios.append(ret_tokens / doc_tokens if doc_tokens > 0 else 0.0)
                if gt_lower and gt_lower in retrieved.lower():
                    hits += 1
            except Exception:
                cost_ratios.append(0.0)
        stats[rule_name] = {
            "hit_rate": round(hits / n, 4),
            "avg_cost_ratio": round(sum(cost_ratios) / len(cost_ratios), 5) if cost_ratios else 0.0,
        }

    return stats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def rule_gen_agent(
    documents: list[dict],
    question: str,
    ground_truth: dict,
    model_name: str = "gpt54",
    rules_dir: str = "rules/gpt54/financebench",
    output_dir: str = "results/financebench/rule_gen",
    logs_dir: str = "logs/financebench/agent",
    max_iterations: int = 12,
) -> dict:
    """Use a LangChain agent with a structured tool set to iteratively generate,
    validate, and refine span-retrieval rules. Returns a result dict matching
    the rule_gen_llm_coarse output schema, with additional agent metadata.
    """
    question_slug = _make_question_slug(question)
    n = len(documents)
    now = datetime.now(timezone.utc)
    timestamp_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    timestamp_file = now.strftime("%Y%m%d_%H%M%S")
    model_mod = importlib.import_module(f"models.{model_name}")

    # Shared mutable state — captured by tool closures
    registered_rules: dict[str, tuple[str, str]] = {}
    total_llm_calls: list[int] = [0]
    judge_tokens: dict = {"input": 0, "output": 0}

    # Build tools
    tools = [
        _make_summarize_answer_locations_tool(documents, ground_truth),
        _make_inspect_answer_context_tool(documents, ground_truth),
        _make_write_rule_tool(registered_rules),
        _make_test_rule_tool(documents, ground_truth, registered_rules),
        _make_show_uncovered_docs_tool(documents, ground_truth, registered_rules),
        _make_diagnose_failing_doc_tool(documents, ground_truth, registered_rules),
        _make_test_union_tool(
            documents, question, ground_truth, model_mod, registered_rules,
            total_llm_calls, judge_tokens,
        ),
    ]

    llm = _azure_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", _AGENT_SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
    )

    user_message = _build_user_prompt(documents, question, ground_truth)

    t0 = time.monotonic()
    agent_result = run_agent(executor, user_message)
    latency_seconds = time.monotonic() - t0

    token_usage = agent_result.get("token_usage", {})
    tool_calls: list[dict] = agent_result.get("tool_calls", [])
    final_output: str = agent_result.get("output", "")

    # Also extract any rules the agent wrote in final prose
    if final_output:
        for func_name, description, source in _extract_functions(final_output):
            if func_name not in registered_rules:
                ns: dict = {}
                try:
                    exec(compile(source, f"<{func_name}>", "exec"), ns)
                    fn = ns.get(func_name)
                    if callable(fn):
                        registered_rules[func_name] = (source, description)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Final merge evaluation
    # ------------------------------------------------------------------
    all_rule_fns = _compile_registered(registered_rules)
    if all_rule_fns:
        merge_report = _run_rules_on_docs(
            all_rule_fns, documents, question, ground_truth, model_mod,
            union_mode=True, judge_tokens=judge_tokens,
        )
        total_llm_calls[0] += 2 * n
        m = merge_report.get("_merge", {})
        merge_accuracy = m.get("accuracy", 0.0)
        merge_num_correct = m.get("num_correct", 0)
        avg_cost_ratio = m.get("avg_cost_ratio", 0.0)
        merge_per_doc = {
            dn: {
                "predicted": dr.get("predicted"),
                "ground_truth": dr.get("ground_truth"),
                "correct": dr.get("correct"),
                "cost_ratio": dr.get("cost_ratio"),
            }
            for dn, dr in m.get("per_doc", {}).items()
        }
    else:
        merge_accuracy = 0.0
        merge_num_correct = 0
        avg_cost_ratio = 0.0
        merge_per_doc = {}

    # ------------------------------------------------------------------
    # Compute per-rule stats (free, no LLM)
    # ------------------------------------------------------------------
    rule_stats = _compute_rule_stats(registered_rules, documents, ground_truth)

    # ------------------------------------------------------------------
    # Save rule files to {rules_dir}/{question_slug}_{n}_llm/
    # ------------------------------------------------------------------
    rule_subdir = Path(rules_dir) / f"{question_slug}_{n}_llm"
    rule_subdir.mkdir(parents=True, exist_ok=True)

    rules_list: list[dict] = []
    for rule_name, (source, description) in registered_rules.items():
        rule_file = rule_subdir / f"{rule_name}.py"
        rule_file.write_text(source + "\n", encoding="utf-8")
        rs = rule_stats.get(rule_name, {"hit_rate": 0.0, "avg_cost_ratio": 0.0})
        rules_list.append({
            "rule_name": rule_name,
            "description": description,
            "hit_rate": rs["hit_rate"],
            "avg_cost_ratio": rs["avg_cost_ratio"],
            "file": str(rule_file),
        })

    # ------------------------------------------------------------------
    # Trace log
    # ------------------------------------------------------------------
    log_path = Path(logs_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    agent_input_tokens  = token_usage.get("prompt_tokens", 0)
    agent_output_tokens = token_usage.get("completion_tokens", 0)
    judge_input_tokens  = judge_tokens["input"]
    judge_output_tokens = judge_tokens["output"]
    total_input_tokens  = agent_input_tokens + judge_input_tokens
    total_output_tokens = agent_output_tokens + judge_output_tokens
    total_llm_calls_count = total_llm_calls[0]

    trace_lines: list[str] = [
        f"Question: {question}",
        f"Slug: {question_slug}",
        f"Docs (N): {n}",
        f"Latency: {latency_seconds:.3f}s",
        f"Agent turns (tool calls): {len(tool_calls)}",
        f"Agent input tokens: {agent_input_tokens}",
        f"Agent output tokens: {agent_output_tokens}",
        f"Judge input tokens: {judge_input_tokens}",
        f"Judge output tokens: {judge_output_tokens}",
        f"Total LLM calls (QA+judge): {total_llm_calls_count}",
        f"Merge accuracy: {merge_num_correct}/{n} ({merge_accuracy:.2f})",
        f"Avg cost ratio: {avg_cost_ratio:.5f}",
        "",
    ]
    for i, tc in enumerate(tool_calls, 1):
        tname = tc.get("name", "?")
        inp = tc.get("input") or {}
        out = tc.get("output", "") or ""
        trace_lines.append(f"=== Step {i}: {tname} ===")
        try:
            inp_str = json.dumps(inp, ensure_ascii=False)
        except Exception:
            inp_str = str(inp)
        trace_lines.append(f"Input: {inp_str}")
        trace_lines.append(f"Output:\n{out}")
        trace_lines.append("")

    trace_lines.append("=== Registered Rules ===")
    for r in rules_list:
        trace_lines.append(
            f"  {r['rule_name']:<50}  hit_rate={r['hit_rate']:.2f}  "
            f"avg_cost={r['avg_cost_ratio']:.5f}  file={r['file']}"
        )
    trace_lines.append("")
    trace_lines.append(
        f"Merge accuracy:        {merge_accuracy:.2f}  ({merge_num_correct}/{n})\n"
        f"Avg cost ratio:        {avg_cost_ratio:.5f}\n"
        f"Total LLM calls:       {total_llm_calls_count}\n"
        f"Agent input tokens:    {agent_input_tokens}\n"
        f"Agent output tokens:   {agent_output_tokens}\n"
        f"Judge input tokens:    {judge_input_tokens}\n"
        f"Judge output tokens:   {judge_output_tokens}\n"
        f"Latency:               {latency_seconds:.1f}s"
    )
    trace_lines.append("")

    trace_lines.append("=== Final Answer ===")
    trace_lines.append(final_output or "(no final message)")

    trace_file = log_path / f"{question_slug}_{n}docs_{timestamp_file}.txt"
    trace_file.write_text("\n".join(trace_lines), encoding="utf-8")

    # ------------------------------------------------------------------
    # Result JSON
    # ------------------------------------------------------------------
    doc_names = [
        d.get("doc_name", d.get("origin", {}).get("filename", "unknown"))
        for d in documents
    ]
    result: dict[str, Any] = {
        "question": question,
        "question_slug": question_slug,
        "timestamp": timestamp_iso,
        "model": model_name,
        "num_documents": n,
        "doc_names": doc_names,
        "max_iterations": max_iterations,
        "agent_turns": len(tool_calls),
        "latency_seconds": round(latency_seconds, 3),
        "agent_input_tokens": agent_input_tokens,
        "agent_output_tokens": agent_output_tokens,
        "judge_input_tokens": judge_input_tokens,
        "judge_output_tokens": judge_output_tokens,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_llm_calls": total_llm_calls_count,
        "merge_accuracy": merge_accuracy,
        "merge_num_correct": merge_num_correct,
        "avg_cost_ratio": avg_cost_ratio,
        "merge_per_doc": merge_per_doc,
        "rules": rules_list,
        "rule_dir": str(rule_subdir),
        "log_file": str(trace_file),
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / f"{question_slug}_{n}_{timestamp_file}.json"
    result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result
