"""Run the FinanceBench 30-question suite against a reconstructed JSON document.

Looks for data/financebench/processing/3M_2017_10K_reconstructed.json first;
if missing, uses the first available *_reconstructed.json in that directory.
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
# Pydantic / AzureChatOpenAI compat fix (mirrors agent.py)
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
# Credential loading (no azure_local dependency)
# ---------------------------------------------------------------------------
def _load_credentials(azure_json_path: Path) -> dict[str, str]:
    cfg: dict[str, str] = json.loads(azure_json_path.read_text())
    key_file = cfg.get("key_file", "")
    if key_file:
        for line in Path(key_file).read_text().splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                cfg[k.strip()] = v.strip()
    return cfg


_creds = _load_credentials(_ROOT / "local" / "azure.json")
AZURE_ENDPOINT = _creds.get("azure_endpoint", "")
AZURE_API_KEY = _creds.get("api_key", "")
AZURE_API_VERSION = _creds.get("api_version", "2024-12-01-preview")
AZURE_DEPLOYMENT = _creds.get("deployment", _creds.get("model_name", "gpt-5.4"))

# ---------------------------------------------------------------------------
# Document selection
# ---------------------------------------------------------------------------
PROCESSING_DIR = _ROOT / "data" / "financebench" / "processing"
_preferred = PROCESSING_DIR / "3M_2017_10K_reconstructed.json"

if _preferred.exists():
    TARGET_JSON = _preferred
else:
    candidates = sorted(PROCESSING_DIR.glob("*_reconstructed.json"))
    if not candidates:
        print("ERROR: No *_reconstructed.json files found under", PROCESSING_DIR, file=sys.stderr)
        sys.exit(1)
    TARGET_JSON = candidates[0]
    print(f"[info] 3M_2017_10K_reconstructed.json not found; using {TARGET_JSON.name}")

with TARGET_JSON.open(encoding="utf-8") as _f:
    _DOC = json.load(_f)

DOC_NAME: str = _DOC["doc_name"]
TEXTS: list[dict] = _DOC["texts"]

print(f"[info] Loaded {DOC_NAME} — {len(TEXTS)} spans")

# ---------------------------------------------------------------------------
# Inline tools (no src/tools imports)
# ---------------------------------------------------------------------------
_LEVEL_ORDER = {"H1": 0, "H2": 1, "H3": 2, "H4": 3, "Body": 4}


@tool
def search_spans(keyword: str, max_results: int = 15) -> str:
    """Search all document spans for a keyword (case-insensitive).

    Returns matching spans with page number, label, and section path.
    """
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
    return "\n\n---\n\n".join(hits) if hits else f"No spans found containing '{keyword}'."


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
    parts: list[str] = []
    for s in spans:
        parts.append(f"[{s.get('label','text')}] {s['text']}")
    return "\n\n".join(parts)


@tool
def get_section_content(section_name: str, max_chars: int = 5000) -> str:
    """Return all spans under the first section header whose text contains section_name.

    Stops at the next header of equal or higher rank.
    """
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

        # already in section — stop at sibling or ancestor header
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
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = f"""You are a precise financial document analyst. You are answering questions about a SEC filing.

Document: {DOC_NAME}  |  Total spans: {len(TEXTS)}

The document is a structured JSON of text spans. Each span has: text, label \
(text / section_header / table / list_item), page_no, and structure.path_text \
(hierarchical section path).

Tools available:
- search_spans(keyword, max_results): keyword search across all spans.
- list_section_headers(): full section hierarchy / table of contents.
- get_page_content(page_no): all content on a page.
- get_section_content(section_name): text under a named section (e.g. "Item 1A").

Answer strategy:
1. Cover-page facts (name, ticker, address, shares): get_page_content(1) or search_spans.
2. Financial line items (revenue, net income, total assets): search_spans with exact term.
3. Section questions (Item 1A risk factors, Item 7 MD&A, etc.): get_section_content.
4. Always cite the page number in your answer.
5. Be concise and direct. If the information is not in the document, say "not found".
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
        max_iterations=20,
        handle_parsing_errors=True,
    )


def _extract_usage(msg: Any) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) from an AIMessage."""
    um = getattr(msg, "usage_metadata", None)
    if isinstance(um, dict):
        return int(um.get("input_tokens") or 0), int(um.get("output_tokens") or 0)
    rm = getattr(msg, "response_metadata", {}) or {}
    tu = rm.get("token_usage") if isinstance(rm, dict) else None
    if isinstance(tu, dict):
        return int(tu.get("prompt_tokens") or 0), int(tu.get("completion_tokens") or 0)
    return 0, 0


# ---------------------------------------------------------------------------
# Run one question (async, using astream_events for token counts)
# ---------------------------------------------------------------------------
async def _arun_question(question: str, q_idx: int, log_dir: Path) -> dict[str, Any]:
    executor = _make_executor()

    state: dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "tool_records": [],
        "final_output": None,
        "last_ai": None,
        "num_iterations": 0,
    }

    async for event in executor.astream_events({"input": question}, version="v2"):
        et = event.get("event")
        name = event.get("name", "")
        data = event.get("data") or {}

        if et == "on_tool_start":
            state["tool_records"].append({
                "tool": name,
                "input": data.get("input"),
                "output": None,
            })
        elif et == "on_tool_end":
            if state["tool_records"]:
                state["tool_records"][-1]["output"] = str(data.get("output", ""))[:3000]
            state["num_iterations"] += 1
        elif et == "on_chat_model_end":
            msg = data.get("output")
            it, ot = _extract_usage(msg)
            state["input_tokens"] += it
            state["output_tokens"] += ot
            if msg is not None:
                c = getattr(msg, "content", None)
                if isinstance(c, str) and c.strip():
                    state["last_ai"] = c
        elif et == "on_chain_end":
            out = data.get("output")
            if isinstance(out, dict) and "output" in out:
                state["final_output"] = out["output"]

    output: str = state["final_output"] or state["last_ai"] or ""
    tool_records: list[dict] = state["tool_records"]
    tools_used = list(dict.fromkeys(r["tool"] for r in tool_records))
    num_iterations = state["num_iterations"]

    return output, tool_records, tools_used, num_iterations, state["input_tokens"], state["output_tokens"]


def _run_question(question: str, q_idx: int, log_dir: Path) -> dict[str, Any]:
    t0 = time.time()
    output, tool_records, tools_used, num_iterations, input_tokens, output_tokens = asyncio.run(
        _arun_question(question, q_idx, log_dir)
    )
    elapsed = time.time() - t0

    # write trace
    trace: list[str] = [
        f"Question {q_idx}: {question}",
        f"Document: {DOC_NAME}",
        f"Latency: {elapsed:.3f}s",
        f"Iterations: {num_iterations}",
        f"Input tokens: {input_tokens}",
        f"Output tokens: {output_tokens}",
        "",
    ]
    for i, rec in enumerate(tool_records, 1):
        trace.append(f"=== Step {i}: {rec['tool']} ===")
        trace.append(f"Input: {json.dumps(rec['input'], ensure_ascii=False)}")
        trace.append(f"Retrieved content:\n{rec.get('output', '')}")
        trace.append("")
    trace += ["=== Final Answer ===", output]

    log_path = log_dir / f"{DOC_NAME}_{q_idx}_trace.txt"
    log_path.write_text("\n".join(trace), encoding="utf-8")

    return {
        "predicted_answer": output,
        "latency_seconds": round(elapsed, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tools_used": tools_used,
        "num_iterations": num_iterations,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    queries_path = _ROOT / "data" / "financebench" / "queries.txt"
    questions = [l.strip() for l in queries_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    log_dir = _ROOT / "logs"
    results_dir = _ROOT / "results"
    log_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    pdf_key = f"{DOC_NAME}.pdf"
    all_results: dict[str, Any] = {pdf_key: {}}

    for idx, question in enumerate(questions, 1):
        print(f"[{idx:02d}/{len(questions)}] {question[:90]}")
        try:
            rec = _run_question(question, idx, log_dir)
        except Exception as exc:
            rec = {
                "predicted_answer": f"ERROR: {exc}",
                "latency_seconds": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "tools_used": [],
                "num_iterations": 0,
            }
            print(f"  ERROR: {exc}")
        all_results[pdf_key][question] = rec
        ans_preview = rec["predicted_answer"][:100].replace("\n", " ")
        print(f"  -> {ans_preview}  ({rec['latency_seconds']}s, {rec['num_iterations']} steps)")

    out_path = results_dir / f"{DOC_NAME}_results.json"
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults -> {out_path}")
    print(f"Traces  -> {log_dir}/{DOC_NAME}_*_trace.txt")


if __name__ == "__main__":
    main()
