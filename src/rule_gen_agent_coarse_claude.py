"""Rule generation via iterative agent with LLM-as-judge evaluation.

Improvements over rule_gen_agent_coarse.py:
- Uses LLM-as-judge (not strict string match) so the agent sees realistic accuracy
- Adds inspect_doc tool so the agent can explore document structure before writing rules
- test_rules evaluates the UNION of all submitted rules (merge accuracy) in addition
  to per-rule accuracy, giving the agent a single target metric to maximise
- Debug output in test_rules shows retrieved text and predicted answer per failing doc
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
# System prompts
# ---------------------------------------------------------------------------

_AGENT_SYSTEM_PROMPT = """\
You are a document rule engineer. Your task is to write Python functions that
locate the text span(s) containing the answer to a question inside financial
document JSONs.

DOCUMENT STRUCTURE
Each document is a dict with a "texts" list of span dicts:
  text            — raw string content (tables in Markdown pipe format)
  label           — "text" | "section_header" | "table" | "list_item"
  page_no         — integer
  bold            — 1 if bold, else 0
  size            — font size in points
  structure.level — "H1" | "H2" | "H3" | "H4" | "Body"
  structure.path_text — breadcrumb of ancestor section headers, pipe-separated
                        e.g. "COMPANY | PART II | Item 8. Financial Statements"
  table_data.cells — list of {{row, col, text, is_column_header, is_row_header}}
                     (only present when label == "table")

RULE INTERFACE
  def rule_<name>(doc: dict) -> list[dict]:
      \"\"\"One-line description.\"\"\"
      ...
      return [span, ...]   # spans from doc["texts"], or synthetic dicts

Rules must be self-contained (import re/json inside if needed).
Never raise — return [] on any failure.
Synthetic spans (dicts not in doc["texts"]) are fine for hint-only cases.

YOUR WORKFLOW
1. Call inspect_doc to study the structure of one or more documents —
   look at path_text patterns, which pages tables appear on, row labels, etc.
2. Write one or more candidate rule_* functions based on what you observe.
3. Call test_rules to evaluate them. The tool reports:
   - per-rule individual accuracy (LLM-judged)
   - MERGE accuracy = accuracy when ALL submitted rules are unioned together
   - For each failing doc: what text was retrieved and what answer was predicted
4. Use the failure details to refine rules or add new rules covering the gaps.
5. Repeat until merge accuracy >= 0.90, or you cannot improve further.

STRATEGY HINTS
- Look at structure.path_text to find reliable section anchors.
- Prefer tables from Item 6 / Item 8 / Notes sections over MD&A narrative.
- If a 8-K filing has no financial statements, return a synthetic span saying "0".
- Exclude noisy sub-sections (Item 7A, Restructuring plans, Item 16 summaries).
- When a balance sheet labels long-term debt as just "Debt", match "| debt |".
- Check multiple docs before writing rules — patterns must generalise."""

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
# Helpers
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


def _run_rules_on_docs(
    rule_fns: dict[str, Any],
    documents: list[dict],
    question: str,
    ground_truth: dict,
    model_mod: Any,
    *,
    union_mode: bool = False,
) -> dict:
    """
    Evaluate rule functions. If union_mode=True, evaluate the union of ALL rules
    as a single combined retrieval. Otherwise evaluate each rule independently.
    Returns a report dict.
    """
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

            # Deduplicate preserving order
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
                # Only show retrieved text for failing docs to keep output compact
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
# Tool factories
# ---------------------------------------------------------------------------

def _make_inspect_doc_tool(documents: list[dict]) -> Any:
    _docs_by_name: dict[str, dict] = {}
    for d in documents:
        name = d.get("doc_name", "unknown")
        filename = d.get("origin", {}).get("filename", name + ".pdf")
        _docs_by_name[name] = d
        _docs_by_name[filename] = d

    doc_names = [d.get("doc_name", "?") for d in documents]

    @tool
    def inspect_doc(
        doc_name: str,
        label_filter: str = "",
        keyword: str = "",
        page: int = 0,
        max_spans: int = 30,
    ) -> str:
        """Inspect spans in a specific document to understand its structure.

        Args:
            doc_name:     Document name (from the list provided). Use "ALL" to sample
                          from all docs (first 5 matching spans each).
            label_filter: If set, only return spans where label == this value
                          (e.g. "table", "section_header").
            keyword:      If set, only return spans whose text contains this keyword
                          (case-insensitive).
            page:         If > 0, filter to this page number only.
            max_spans:    Maximum number of spans to return (default 30).

        Returns:
            JSON list of span summaries: {doc_name, page_no, label, path_text, text_preview}.
        """
        target_docs: list[dict]
        if doc_name.upper() == "ALL":
            target_docs = documents
        else:
            doc = _docs_by_name.get(doc_name)
            if doc is None:
                return json.dumps({
                    "error": f"Unknown doc_name {doc_name!r}",
                    "available": doc_names,
                })
            target_docs = [doc]

        results: list[dict] = []
        per_doc_count: dict[str, int] = {}
        limit_per_doc = max_spans if doc_name.upper() != "ALL" else max(1, max_spans // len(documents))

        for doc in target_docs:
            dname = doc.get("doc_name", "unknown")
            count = 0
            for s in doc.get("texts", []):
                if count >= limit_per_doc:
                    break
                if label_filter and s.get("label") != label_filter:
                    continue
                if page > 0 and s.get("page_no") != page:
                    continue
                text = s.get("text") or ""
                if keyword and keyword.lower() not in text.lower():
                    continue
                path = (s.get("structure") or {}).get("path_text", "")
                results.append({
                    "doc_name": dname,
                    "page_no": s.get("page_no"),
                    "label": s.get("label"),
                    "path_text": path,
                    "text_preview": text[:300],
                })
                count += 1

        return json.dumps(results, indent=2)

    return inspect_doc


def _make_test_rules_tool(
    documents: list[dict],
    question: str,
    ground_truth: dict,
    model_mod: Any,
    accumulated_rules: dict[str, tuple[str, str]],
) -> Any:
    """
    accumulated_rules is mutated across calls: the agent's approved rules accumulate here.
    Each call to test_rules adds newly submitted rules to the accumulation.
    """
    _docs = documents
    _question = question
    _gt = ground_truth
    _mod = model_mod
    _acc = accumulated_rules

    @tool
    def test_rules(code: str) -> str:
        """Evaluate rule_* functions against all sample documents using LLM-as-judge.

        Accepts Python source defining one or more rule_* functions. Evaluates:
          1. Each rule INDIVIDUALLY — to see which docs each rule covers.
          2. MERGE of ALL previously submitted rules PLUS new rules — the combined
             union accuracy, which is the primary target metric.

        For failing documents, shows the retrieved text preview and predicted answer
        so you can diagnose why the rule is wrong.

        Returns JSON with keys:
          individual: {rule_name: {accuracy, num_correct, per_doc: {...}}}
          merge:      {accuracy, num_correct, avg_cost_ratio, per_doc: {...}}

        Aim to maximise merge.accuracy (target >= 0.90).
        """
        # Compile submitted code
        ns: dict = {}
        try:
            exec(compile(code, "<test_rules>", "exec"), ns)
        except Exception as exc:
            return json.dumps({"error": f"Compilation error: {exc}"})

        new_rule_fns = {k: v for k, v in ns.items() if k.startswith("rule_") and callable(v)}
        if not new_rule_fns:
            return json.dumps({"error": "No rule_* functions found in submitted code"})

        # Extract sources and accumulate
        new_sources = _extract_functions(code)
        for func_name, description, source in new_sources:
            _acc[func_name] = (source, description)

        # 1. Individual evaluation
        individual_report = _run_rules_on_docs(
            new_rule_fns, _docs, _question, _gt, _mod, union_mode=False
        )

        # 2. Merge evaluation: compile all accumulated rules
        all_rule_fns: dict[str, Any] = {}
        for rname, (source, _) in _acc.items():
            try:
                rns: dict = {}
                exec(compile(source, f"<{rname}>", "exec"), rns)
                fn = rns.get(rname)
                if callable(fn):
                    all_rule_fns[rname] = fn
            except Exception:
                pass

        merge_report = _run_rules_on_docs(
            all_rule_fns, _docs, _question, _gt, _mod, union_mode=True
        )

        result = {
            "individual": individual_report,
            "merge": merge_report.get("_merge", {}),
            "accumulated_rule_names": sorted(_acc.keys()),
        }
        return json.dumps(result, indent=2)

    return test_rules


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------

def _build_user_prompt(documents: list[dict], question: str, ground_truth: dict) -> str:
    n = len(documents)
    parts: list[str] = [
        f"I have {n} financial documents. I need rules to locate the answer to:\n"
        f"QUESTION: {question}\n\n"
        "Ground truth answers (by document filename):\n",
    ]
    for doc in documents:
        doc_name = doc.get("doc_name", "unknown")
        filename = doc.get("origin", {}).get("filename", doc_name + ".pdf")
        answer = ground_truth.get(filename, ground_truth.get(doc_name, "N/A"))
        parts.append(f"  {filename}: {answer}\n")

    parts.append(
        "\nDocument names available for inspect_doc:\n"
        + "\n".join(f"  {d.get('doc_name', '?')}" for d in documents)
    )
    parts.append(
        "\n\nStart by calling inspect_doc on a few documents to understand their "
        "structure before writing rules. Then call test_rules to evaluate. "
        "Iterate until merge accuracy >= 0.90 or you cannot improve further."
    )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Post-run helpers
# ---------------------------------------------------------------------------

def _eval_merge_final(
    rule_functions: dict[str, tuple[str, str]],
    documents: list[dict],
    question: str,
    ground_truth: dict,
    model_mod: Any,
) -> dict:
    """Final merge evaluation with full per-doc detail."""
    all_rule_fns: dict[str, Any] = {}
    for rname, (source, _) in rule_functions.items():
        try:
            ns: dict = {}
            exec(compile(source, f"<{rname}>", "exec"), ns)
            fn = ns.get(rname)
            if callable(fn):
                all_rule_fns[rname] = fn
        except Exception:
            pass

    if not all_rule_fns:
        n = len(documents)
        return {
            "merge_accuracy": 0.0,
            "merge_num_correct": 0,
            "avg_cost_ratio": 0.0,
            "merge_per_doc": {},
        }

    report = _run_rules_on_docs(
        all_rule_fns, documents, question, ground_truth, model_mod, union_mode=True
    )
    m = report.get("_merge", {})

    per_doc_clean: dict = {}
    for doc_name, dr in m.get("per_doc", {}).items():
        per_doc_clean[doc_name] = {
            "predicted": dr.get("predicted"),
            "ground_truth": dr.get("ground_truth"),
            "correct": dr.get("correct"),
            "cost_ratio": dr.get("cost_ratio"),
        }

    return {
        "merge_accuracy": m.get("accuracy", 0.0),
        "merge_num_correct": m.get("num_correct", 0),
        "avg_cost_ratio": m.get("avg_cost_ratio", 0.0),
        "merge_per_doc": per_doc_clean,
    }


def _eval_individual_final(
    rule_name: str,
    source: str,
    documents: list[dict],
    question: str,
    ground_truth: dict,
    model_mod: Any,
) -> dict:
    ns: dict = {}
    try:
        exec(compile(source, f"<{rule_name}>", "exec"), ns)
    except Exception as exc:
        return {"individual_accuracy": 0.0, "individual_num_correct": 0, "error": str(exc)}
    rule_fn = ns.get(rule_name)
    if not callable(rule_fn):
        return {"individual_accuracy": 0.0, "individual_num_correct": 0}

    report = _run_rules_on_docs(
        {rule_name: rule_fn}, documents, question, ground_truth, model_mod, union_mode=False
    )
    r = report.get(rule_name, {})
    return {
        "individual_accuracy": r.get("accuracy", 0.0),
        "individual_num_correct": r.get("num_correct", 0),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def rule_gen_agent(
    documents: list[dict],
    question: str,
    ground_truth: dict,
    model_name: str = "gpt54",
    output_dir: str = "results/financebench/rule_gen_agent",
    rules_dir: str = "rules/financebench",
    logs_dir: str = "logs/financebench/agent",
    max_iterations: int = 15,
    target_accuracy: float = 0.90,
) -> dict:
    """Iteratively generate, test, and refine span-retrieval rules using an LLM agent.

    Key improvements over rule_gen_agent_coarse:
    - LLM-as-judge evaluation so the agent sees realistic accuracy
    - inspect_doc tool for document structure exploration
    - Accumulated rule set across turns (agent builds up a library)
    - Merge accuracy tracked as primary metric
    """
    question_slug = _make_question_slug(question)
    n = len(documents)
    now = datetime.now(timezone.utc)
    timestamp_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    timestamp_file = now.strftime("%Y%m%d_%H%M%S")
    model_mod = importlib.import_module(f"models.{model_name}")

    # Shared accumulator — mutated by test_rules across agent turns
    accumulated_rules: dict[str, tuple[str, str]] = {}

    inspect_doc_tool = _make_inspect_doc_tool(documents)
    test_rules_tool = _make_test_rules_tool(
        documents, question, ground_truth, model_mod, accumulated_rules
    )

    llm = _azure_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", _AGENT_SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, [inspect_doc_tool, test_rules_tool], prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=[inspect_doc_tool, test_rules_tool],
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

    # Also scan final output for any rule definitions the agent wrote in prose
    if final_output:
        for func_name, description, source in _extract_functions(final_output):
            accumulated_rules[func_name] = (source, description)

    rule_functions = accumulated_rules

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    merge_result = _eval_merge_final(rule_functions, documents, question, ground_truth, model_mod)

    # Individual accuracy per rule
    rules_list: list[dict] = []
    for rule_name, (source, description) in rule_functions.items():
        indl = _eval_individual_final(rule_name, source, documents, question, ground_truth, model_mod)
        rules_list.append({
            "rule_name": rule_name,
            "description": description,
            "individual_accuracy": indl["individual_accuracy"],
            "individual_num_correct": indl["individual_num_correct"],
        })

    # ------------------------------------------------------------------
    # Save rule files
    # ------------------------------------------------------------------
    rule_subdir = Path(rules_dir) / f"{question_slug}_{n}_agent_claude"
    rule_subdir.mkdir(parents=True, exist_ok=True)
    for r in rules_list:
        rname = r["rule_name"]
        source, _ = rule_functions[rname]
        rule_file = rule_subdir / f"{rname}.py"
        rule_file.write_text(source + "\n", encoding="utf-8")
        r["file"] = str(rule_file)

    # ------------------------------------------------------------------
    # Trace log
    # ------------------------------------------------------------------
    log_path = Path(logs_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    trace_lines: list[str] = [
        "=== rule_gen_agent ===",
        f"Question:   {question}",
        f"Slug:       {question_slug}",
        f"Docs (N):   {n}",
        f"Timestamp:  {timestamp_iso}",
        f"Max iters:  {max_iterations}",
        f"Target acc: {target_accuracy}",
        "",
    ]
    for i, tc in enumerate(tool_calls, 1):
        tname = tc.get("name", "?")
        inp = tc.get("input") or {}
        trace_lines.append(f"--- Turn {i}: {tname} ---")
        if tname == "test_rules":
            code = inp.get("code", "") if isinstance(inp, dict) else str(inp)
            parsed = _extract_functions(code)
            trace_lines.append(f"  Submitted rules: {[f[0] for f in parsed]}")
            raw_out = tc.get("output", "") or ""
            try:
                out = json.loads(raw_out)
                merge = out.get("merge", {})
                trace_lines.append(
                    f"  Merge accuracy: {merge.get('num_correct', '?')}/{n} "
                    f"({merge.get('accuracy', '?')})"
                )
                for rn, rd in out.get("individual", {}).items():
                    trace_lines.append(
                        f"  {rn}: {rd.get('num_correct', '?')}/{n} "
                        f"(acc={rd.get('accuracy', '?')})"
                    )
            except Exception:
                trace_lines.append(f"  Output: {raw_out[:300]}")
        elif tname == "inspect_doc":
            trace_lines.append(f"  Args: {inp}")
            trace_lines.append(f"  Output: {str(tc.get('output', ''))[:200]}")
        trace_lines.append("")

    if final_output:
        trace_lines.append("--- Final Agent Message ---")
        trace_lines.append(final_output[:1000])
        trace_lines.append("")

    trace_lines.append("=== Final Rules ===")
    for r in rules_list:
        trace_lines.append(
            f"  {r['rule_name']}: indl={r['individual_accuracy']:.2f} "
            f"({r['individual_num_correct']}/{n})"
        )
    trace_lines.append(
        f"  MERGE: {merge_result['merge_num_correct']}/{n} "
        f"(acc={merge_result['merge_accuracy']:.2f}) "
        f"cost={merge_result['avg_cost_ratio']:.5f}"
    )
    trace_lines.append("")
    trace_lines.append("=== Summary ===")
    trace_lines.append(f"  Agent turns:   {len(tool_calls)}")
    trace_lines.append(f"  Input tokens:  {token_usage.get('prompt_tokens', 0)}")
    trace_lines.append(f"  Output tokens: {token_usage.get('completion_tokens', 0)}")
    trace_lines.append(f"  Latency:       {latency_seconds:.1f}s")

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
        "target_accuracy": target_accuracy,
        "agent_turns": len(tool_calls),
        "latency_seconds": round(latency_seconds, 3),
        "input_tokens": token_usage.get("prompt_tokens", 0),
        "output_tokens": token_usage.get("completion_tokens", 0),
        "merge_accuracy": merge_result["merge_accuracy"],
        "merge_num_correct": merge_result["merge_num_correct"],
        "avg_cost_ratio": merge_result["avg_cost_ratio"],
        "merge_per_doc": merge_result["merge_per_doc"],
        "rules": rules_list,
        "rule_dir": str(rule_subdir),
        "log_file": str(trace_file),
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / f"{question_slug}_{n}_{timestamp_file}.json"
    result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result


# ---------------------------------------------------------------------------
# Quick test entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    os.chdir(_ROOT)

    _DATA = _ROOT / "data" / "financebench"
    LABELS_FILE = _DATA / "sample_doc_labels.json"
    PROCESSING_DIR = _DATA / "processing"
    QUERIES_FILE = _DATA / "sample_queries.txt"

    labels: dict[str, dict] = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    queries = QUERIES_FILE.read_text(encoding="utf-8").splitlines()

    # Default: first query (long-term debt — lowest LLM accuracy at 0.50)
    import sys as _sys
    query_idx = int(_sys.argv[1]) if len(_sys.argv) > 1 else 0
    question = next(
        (q.strip() for i, q in enumerate(queries) if q.strip() and i == query_idx),
        queries[0].strip(),
    )

    docs: list[dict] = []
    for pdf_key in labels:
        stem = Path(pdf_key).stem
        doc_path = PROCESSING_DIR / f"{stem}_reconstructed.json"
        if doc_path.exists():
            docs.append(json.loads(doc_path.read_text(encoding="utf-8")))

    gt: dict[str, Any] = {
        pdf_key: labels[pdf_key][question]
        for pdf_key in labels
        if question in labels[pdf_key]
    }

    print(f"Question    : {question}")
    print(f"Documents   : {len(docs)}")
    print(f"GT entries  : {len(gt)}")
    print()

    result = rule_gen_agent(docs, question, gt)

    print("\n=== Result Summary ===")
    print(f"Agent turns    : {result['agent_turns']}")
    print(f"Rules          : {len(result['rules'])}")
    print(f"Merge accuracy : {result['merge_accuracy']:.2f}  ({result['merge_num_correct']}/{result['num_documents']})")
    print(f"Avg cost ratio : {result['avg_cost_ratio']:.5f}")
    print(f"Latency        : {result['latency_seconds']:.1f}s")
    print(f"Input tokens   : {result['input_tokens']}")
    print(f"Output tokens  : {result['output_tokens']}")
    print()
    print("Rules:")
    for r in result["rules"]:
        print(
            f"  {r['rule_name']:<50}  "
            f"indl={r['individual_accuracy']:.2f} "
            f"({r['individual_num_correct']}/{result['num_documents']})"
        )
    print(f"\nRule dir: {result['rule_dir']}")
    print(f"Log:      {result['log_file']}")
