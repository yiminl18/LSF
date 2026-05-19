"""Baseline strategy 2: Codex-style agentic QA using gpt54 with tool calling.

Mirrors the Codex CLI pattern: gpt54 drives an inner agentic loop with document
tools (read_page, search_spans, get_doc_info). Claude Code is the outer
orchestrator (picks which question/doc, writes results). All reasoning and
iteration decisions are made by gpt54 inside the loop.

Usage (single pair):
    python src/baseline/codex_gpt54_qa.py \
        --doc data/financebench/processing/JPMORGAN_2023_10K_reconstructed.json \
        --question "What is the registrant's telephone number?"
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

_SYSTEM = """\
You are a financial document QA assistant with tools to read a document.
Use the tools to locate the answer to the question, then respond with only the answer string.
Be concise — a number, name, date, address, or short phrase. No explanation.
If after reading you cannot find the answer, respond with: NOT_FOUND"""

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_doc_info",
            "description": "Get document metadata: total pages and total span count.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Read all text spans from a specific page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_no": {"type": "integer", "description": "Page number (1-indexed)"},
                },
                "required": ["page_no"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_spans",
            "description": "Return spans whose text contains the keyword (case-insensitive). Max 30 results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Keyword to search for"},
                },
                "required": ["keyword"],
            },
        },
    },
]


def _get_doc_info(doc: dict) -> str:
    spans = doc.get("texts", [])
    pages = {s.get("page_no", 0) for s in spans}
    return json.dumps({"total_pages": max(pages) if pages else 0, "total_spans": len(spans)})


def _read_page(doc: dict, page_no: int) -> str:
    spans = [s for s in doc.get("texts", []) if s.get("page_no") == page_no]
    out = [{"text": s.get("text", ""), "bold": s.get("bold", False),
            "size": s.get("size"), "label": s.get("label")} for s in spans]
    return json.dumps(out)


def _search_spans(doc: dict, keyword: str) -> str:
    kw = keyword.lower()
    hits = [s for s in doc.get("texts", []) if kw in s.get("text", "").lower()][:30]
    out = [{"text": s.get("text", ""), "page_no": s.get("page_no"),
            "bold": s.get("bold", False), "label": s.get("label")} for s in hits]
    return json.dumps(out)


def _dispatch(tool_name: str, args: dict, doc: dict) -> str:
    if tool_name == "get_doc_info":
        return _get_doc_info(doc)
    if tool_name == "read_page":
        return _read_page(doc, args["page_no"])
    if tool_name == "search_spans":
        return _search_spans(doc, args["keyword"])
    return json.dumps({"error": f"unknown tool: {tool_name}"})


def run_qa(doc_path: str | Path, question: str, model: str = "gpt54",
           max_iterations: int = 10, **_) -> dict:
    gpt54 = importlib.import_module("models.gpt54")
    doc   = json.loads(Path(doc_path).read_text(encoding="utf-8"))

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": f"Question: {question}"},
    ]

    total_input = total_output = iterations = tool_calls_n = 0
    t0 = time.time()
    answer = None

    for _ in range(max_iterations):
        iterations += 1
        resp = gpt54.client.chat.completions.create(
            model=gpt54.AZURE_DEPLOYMENT,
            messages=messages,
            tools=_TOOLS,
            tool_choice="auto",
            temperature=0.0,
        )
        total_input  += resp.usage.prompt_tokens     if resp.usage else 0
        total_output += resp.usage.completion_tokens if resp.usage else 0

        msg = resp.choices[0].message
        messages.append({"role": "assistant", "content": msg.content,
                         "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])]})

        if not msg.tool_calls:
            answer = (msg.content or "").strip()
            break

        # Execute all tool calls in this turn
        for tc in msg.tool_calls:
            tool_calls_n += 1
            args   = json.loads(tc.function.arguments or "{}")
            result = _dispatch(tc.function.name, args, doc)
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

    return {
        "status":          "ok",
        "answer":          answer,
        "input_tokens":    total_input,
        "output_tokens":   total_output,
        "latency_seconds": round(time.time() - t0, 2),
        "total_cost_usd":  None,
        "iterations":      iterations,
        "tool_calls":      tool_calls_n,
        "model":           gpt54.AZURE_DEPLOYMENT,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Codex-style gpt54 agentic QA — single pair")
    ap.add_argument("--doc",            required=True)
    ap.add_argument("--question",       required=True)
    ap.add_argument("--max-iterations", type=int, default=10)
    args = ap.parse_args()

    result = run_qa(args.doc, args.question, max_iterations=args.max_iterations)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
