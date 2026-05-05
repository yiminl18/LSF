"""Verify ground truth answers for a FinanceBench document using an agent.

Usage:
  python verify_financebench.py [DOC_NAME]

DOC_NAME defaults to ADOBE_2022Q2_10Q if not supplied.
The document must have a matching *_reconstructed.json and *.txt_answers.json.

Outputs:
  data/financebench/truth_analysis/{DOC_NAME}_ground_truth_analysis.txt
  data/financebench/correct_labels.json   — updated with new entry
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Pydantic / AzureChatOpenAI compat fix
# ---------------------------------------------------------------------------
from langchain_openai import AzureChatOpenAI

try:
    import langchain_core.language_models.base as _lm_base
    import langchain_openai.chat_models.azure as _azure_mod
    import langchain_openai.chat_models.base as _openai_base
    from langchain_core.caches import BaseCache as _BaseCache
    from langchain_core.callbacks import Callbacks as _Callbacks
    from langchain_core.outputs import LLMResult as _LLMResult

    _ns = {"BaseCache": _BaseCache, "Callbacks": _Callbacks, "LLMResult": _LLMResult}
    for _mod in (_lm_base, _azure_mod, _openai_base):
        _mod.BaseCache = _BaseCache  # type: ignore[attr-defined]
        _mod.Callbacks = _Callbacks  # type: ignore[attr-defined]
        _mod.LLMResult = _LLMResult  # type: ignore[attr-defined]
    AzureChatOpenAI.model_rebuild(_types_namespace=_ns)
except Exception:
    pass

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
def _load_credentials(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = json.loads(path.read_text())
    key_file = cfg.get("key_file", "")
    if key_file:
        for line in Path(key_file).read_text().splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                cfg[k.strip()] = v.strip()
    return cfg

_creds = _load_credentials(_ROOT / "local" / "azure.json")
AZURE_ENDPOINT   = _creds.get("azure_endpoint", "")
AZURE_API_KEY    = _creds.get("api_key", "")
AZURE_API_VERSION = _creds.get("api_version", "2024-12-01-preview")
AZURE_DEPLOYMENT  = _creds.get("deployment", _creds.get("model_name", "gpt-5.4"))

# ---------------------------------------------------------------------------
# Document  (can be overridden via CLI arg)
# ---------------------------------------------------------------------------
DOC_NAME   = sys.argv[1] if len(sys.argv) > 1 else "ADOBE_2022Q2_10Q"
JSON_PATH  = _ROOT / "data" / "financebench" / "processing" / f"{DOC_NAME}_reconstructed.json"
GT_PATH    = _ROOT / "data" / "financebench" / "ground_truth" / f"{DOC_NAME}.txt_answers.json"
QUERIES_PATH = _ROOT / "data" / "financebench" / "queries.txt"

with JSON_PATH.open(encoding="utf-8") as _f:
    _DOC = json.load(_f)
TEXTS: list[dict] = _DOC["texts"]
print(f"[info] Loaded {DOC_NAME} — {len(TEXTS)} spans")

with GT_PATH.open(encoding="utf-8") as _f:
    GT: dict[str, Any] = json.load(_f)   # keys "1".."30"

QUESTIONS = [l.strip() for l in QUERIES_PATH.read_text().splitlines() if l.strip()]

# ---------------------------------------------------------------------------
# Inline document tools (no src/tools imports)
# ---------------------------------------------------------------------------
_LEVEL_ORDER = {"H1": 0, "H2": 1, "H3": 2, "H4": 3, "Body": 4}


@tool
def search_spans(keyword: str, max_results: int = 15) -> str:
    """Search all document spans for a keyword (case-insensitive).
    Returns matching spans with page number, label, and section path."""
    kw = keyword.lower()
    hits: list[str] = []
    for span in TEXTS:
        if kw in span.get("text", "").lower():
            page = span.get("page_no", "?")
            label = span.get("label", "text")
            path = span.get("structure", {}).get("path_text", "")
            hits.append(f"[page {page}][{label}][{path}]\n{span['text']}")
            if len(hits) >= max_results:
                break
    return "\n\n---\n\n".join(hits) if hits else f"No spans found for '{keyword}'."


@tool
def list_section_headers() -> str:
    """Return all section headers in document order with page numbers and heading level."""
    rows: list[str] = []
    for span in TEXTS:
        if span.get("label") == "section_header":
            level = span.get("structure", {}).get("level", "?")
            page = span.get("page_no", "?")
            rows.append(f"[page {page}][{level}] {span['text']}")
    return "\n".join(rows) if rows else "No section headers found."


@tool
def get_page_content(page_no: int) -> str:
    """Return every span on the given page number."""
    spans = [s for s in TEXTS if s.get("page_no") == page_no]
    if not spans:
        return f"No content on page {page_no}."
    return "\n\n".join(f"[{s.get('label','text')}] {s['text']}" for s in spans)


@tool
def get_section_content(section_name: str, max_chars: int = 5000) -> str:
    """Return all spans under the first section header whose text contains section_name."""
    name = section_name.lower()
    in_section = False
    section_rank: int = 99
    parts: list[str] = []
    total = 0
    for span in TEXTS:
        label = span.get("label", "")
        text = span.get("text", "")
        level = span.get("structure", {}).get("level", "Body")
        rank = _LEVEL_ORDER.get(level, 4)
        if not in_section:
            if label == "section_header" and name in text.lower():
                in_section = True
                section_rank = rank
                parts.append(f"[HEADER][page {span.get('page_no','?')}] {text}")
            continue
        if label == "section_header" and rank <= section_rank:
            break
        entry = f"[{label}][page {span.get('page_no','?')}] {text}"
        parts.append(entry)
        total += len(text)
        if total >= max_chars:
            parts.append("... [truncated]")
            break
    return "\n\n".join(parts) if parts else f"No section matching '{section_name}' found."


_TOOLS = [search_spans, list_section_headers, get_page_content, get_section_content]

# ---------------------------------------------------------------------------
# System prompt for verification
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = f"""You are a financial document fact-checker. You are verifying answers about {DOC_NAME}.

The document has {len(TEXTS)} structured spans (text, section_header, table, list_item) with page numbers.

Available tools:
- search_spans(keyword): keyword search across all spans.
- list_section_headers(): full section hierarchy.
- get_page_content(page_no): all content on a page.
- get_section_content(section_name): text under a named section.

For each verification task you will receive:
  QUESTION: the question
  PROPOSED ANSWER: the answer to verify

Your job:
1. Use the tools to find evidence in the document.
2. Compare the evidence to the proposed answer.
3. Respond in this exact format:

VERDICT: CORRECT | INCORRECT | AMBIGUOUS
EVIDENCE: <exact quote or data from the document with page number>
CORRECT_ANSWER: <if INCORRECT or AMBIGUOUS, the answer supported by the document; otherwise repeat the proposed answer>
REASONING: <one or two sentences explaining your verdict>
"""

# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
def _make_executor() -> AgentExecutor:
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        azure_deployment=AZURE_DEPLOYMENT,
        temperature=0.0,
        model_kwargs={"stream_options": {"include_usage": True}},
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, _TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=_TOOLS,
        verbose=False,
        max_iterations=15,
        handle_parsing_errors=True,
    )


def _extract_usage(msg: Any) -> tuple[int, int]:
    um = getattr(msg, "usage_metadata", None)
    if isinstance(um, dict):
        return int(um.get("input_tokens") or 0), int(um.get("output_tokens") or 0)
    rm = getattr(msg, "response_metadata", {}) or {}
    tu = rm.get("token_usage") if isinstance(rm, dict) else None
    if isinstance(tu, dict):
        return int(tu.get("prompt_tokens") or 0), int(tu.get("completion_tokens") or 0)
    return 0, 0


# ---------------------------------------------------------------------------
# Parse structured response from agent
# ---------------------------------------------------------------------------
def _parse_verdict(text: str) -> dict[str, str]:
    result = {"verdict": "", "evidence": "", "correct_answer": "", "reasoning": ""}
    current = None
    buf: list[str] = []

    def flush():
        if current and buf:
            result[current] = " ".join(" ".join(buf).split())

    for line in text.splitlines():
        line = line.strip()
        for key, field in [("VERDICT:", "verdict"), ("EVIDENCE:", "evidence"),
                            ("CORRECT_ANSWER:", "correct_answer"), ("REASONING:", "reasoning")]:
            if line.startswith(key):
                flush()
                buf = [line[len(key):].strip()]
                current = field
                break
        else:
            if current:
                buf.append(line)
    flush()
    return result


# ---------------------------------------------------------------------------
# Verify one question
# ---------------------------------------------------------------------------
async def _averify(question: str, gt_answer: Any) -> dict[str, Any]:
    executor = _make_executor()
    gt_str = json.dumps(gt_answer) if not isinstance(gt_answer, str) else gt_answer

    prompt = (
        f"QUESTION: {question}\n"
        f"PROPOSED ANSWER: {gt_str}\n\n"
        "Use the tools to find evidence, then respond with VERDICT / EVIDENCE / CORRECT_ANSWER / REASONING."
    )

    state: dict[str, Any] = {"output": None, "last_ai": None,
                              "input_tokens": 0, "output_tokens": 0, "iterations": 0}

    async for event in executor.astream_events({"input": prompt}, version="v2"):
        et = event.get("event")
        data = event.get("data") or {}
        if et == "on_tool_end":
            state["iterations"] += 1
        elif et == "on_chat_model_end":
            msg = data.get("output")
            it, ot = _extract_usage(msg)
            state["input_tokens"] += it
            state["output_tokens"] += ot
            c = getattr(msg, "content", None)
            if isinstance(c, str) and c.strip():
                state["last_ai"] = c
        elif et == "on_chain_end":
            out = data.get("output")
            if isinstance(out, dict) and "output" in out:
                state["output"] = out["output"]

    raw = state["output"] or state["last_ai"] or ""
    parsed = _parse_verdict(raw)
    return {**parsed, "raw": raw,
            "input_tokens": state["input_tokens"],
            "output_tokens": state["output_tokens"],
            "iterations": state["iterations"]}


def verify(question: str, gt_answer: Any) -> dict[str, Any]:
    return asyncio.run(_averify(question, gt_answer))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    analysis_lines: list[str] = [
        f"Ground Truth Verification Analysis",
        f"Document: {DOC_NAME}",
        f"Date: 2026-04-27",
        "=" * 60,
        "",
    ]

    corrected_answers: dict[str, Any] = {}
    corrections_found: list[int] = []

    total_it = 0
    total_ot = 0

    for i, question in enumerate(QUESTIONS, 1):
        gt_answer = GT.get(str(i))
        gt_str = json.dumps(gt_answer) if not isinstance(gt_answer, str) else gt_answer

        print(f"[{i:02d}/30] Verifying: {question[:70]}")
        t0 = time.time()
        result = verify(question, gt_answer)
        elapsed = time.time() - t0

        verdict = result.get("verdict", "").upper()
        evidence = result.get("evidence", "")
        correct_answer = result.get("correct_answer", gt_str)
        reasoning = result.get("reasoning", "")

        total_it += result["input_tokens"]
        total_ot += result["output_tokens"]

        # Determine final label to use
        if verdict == "INCORRECT":
            final_answer_raw = correct_answer
            corrections_found.append(i)
            marker = "INCORRECT ✗"
        elif verdict == "AMBIGUOUS":
            final_answer_raw = correct_answer
            corrections_found.append(i)
            marker = "AMBIGUOUS ~"
        else:
            final_answer_raw = gt_answer
            marker = "CORRECT ✓"

        # Store final answer (keep original type if CORRECT, else store corrected string)
        corrected_answers[question] = final_answer_raw if verdict == "CORRECT" else final_answer_raw

        print(f"  {marker}  ({elapsed:.1f}s)")
        if verdict != "CORRECT":
            print(f"  GT:      {gt_str[:80]}")
            print(f"  Correct: {str(correct_answer)[:80]}")

        # Analysis block
        sep = "-" * 60
        analysis_lines += [
            f"Q{i} — {question}",
            sep,
            f"Ground Truth : {gt_str}",
            f"Verdict      : {verdict}",
            f"Evidence     : {evidence}",
            f"Correct Ans  : {correct_answer}",
            f"Reasoning    : {reasoning}",
            "",
        ]

    # Summary
    analysis_lines += [
        "=" * 60,
        "SUMMARY",
        "=" * 60,
        f"Total questions : 30",
        f"Correct         : {30 - len(corrections_found)}",
        f"Incorrect/Ambig : {len(corrections_found)} — Q{', Q'.join(str(n) for n in corrections_found)}",
        f"Input tokens    : {total_it:,}",
        f"Output tokens   : {total_ot:,}",
        f"Est. cost       : ${total_it/1e6*2.5 + total_ot/1e6*15:.4f}",
    ]

    # Write analysis file
    analysis_dir = _ROOT / "data" / "financebench" / "truth_analysis"
    analysis_dir.mkdir(exist_ok=True)
    analysis_path = analysis_dir / f"{DOC_NAME}_ground_truth_analysis.txt"
    analysis_path.write_text("\n".join(analysis_lines), encoding="utf-8")
    print(f"\nAnalysis -> {analysis_path}")

    # Update correct_labels.json
    labels_path = _ROOT / "data" / "financebench" / "correct_labels.json"
    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
    else:
        labels = {}

    pdf_key = f"{DOC_NAME}.pdf"
    labels[pdf_key] = corrected_answers
    labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"correct_labels.json updated with {pdf_key}")
    print(f"\nCorrections: {len(corrections_found)} questions adjusted — Q{corrections_found}")


if __name__ == "__main__":
    main()
