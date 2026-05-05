# Agent Setup — GPT-5.4 via LangChain

This document describes how to configure and launch the document QA agent using a locally-served GPT-5.4 model and the LangChain agent framework.

---

## Data Layout

```
data/financebench/
├── sample.txt              ← list of 50 PDF filenames (one per line)
├── queries.txt             ← 30 fixed questions to run against each document
├── sample_labels.json      ← ground truth: { "filename.pdf": { "question": "answer", ... } }
└── processing/
    ├── 3M_2017_10K_reconstructed.json
    ├── ADOBE_2022Q2_10Q_reconstructed.json
    └── ...                 ← three variants per doc: _reconstructed, _lsf, _docling
```

**Task:** Take the **first line** of `sample.txt` (`3M_2017_10K.pdf`), run all 30 queries from `queries.txt` against its `_reconstructed.json`, and write results + traces.

---

## Document JSON Schema

Each `*_reconstructed.json` file has three top-level keys:

```json
{
  "doc_name": "3M_2017_10K",
  "origin": {
    "mimetype": "application/pdf",
    "binary_hash": 14862264175189151635,
    "filename": "3M_2017_10K.pdf"
  },
  "texts": [ ... ]
}
```

### `texts` array — one entry per text span

| Field | Type | Values / Notes |
|---|---|---|
| `text` | string | Content; tables rendered as Markdown pipe format |
| `label` | string | `"text"`, `"section_header"`, `"table"`, `"list_item"` |
| `page_no` | int | Page number in original PDF |
| `bold` | int | 1 = bold |
| `size` | float | Font size in pts |
| `structure.level` | string | `"H1"`, `"H2"`, `"H3"`, `"H4"`, `"Body"` |
| `structure.path_text` | string | Ancestor breadcrumb, e.g. `"3M COMPANY | Item 1"` |
| `table_data` | object | **Only present when `label == "table"`** |

### `table_data` structure (tables only)

```json
{
  "num_rows": 6,
  "num_cols": 3,
  "cells": [
    {
      "row": 0, "col": 0,
      "text": "Title of each class",
      "row_span": 1, "col_span": 1,
      "is_column_header": true,
      "is_row_header": false
    }
  ]
}
```

---

## Prerequisites

```bash
pip install langchain langchain-community langchain-openai
```

GPT-5.4 local API server must be running at `http://localhost:8000` (started via `python models/gpt54.py`).

---

## Step 1: Start the Local GPT-5.4 API Server

```bash
python models/gpt54.py
```

Verify:

```bash
curl http://localhost:8000/health
```

---

## Step 2: Configure the LangChain LLM

LangChain's `ChatOpenAI` accepts a custom `base_url`, so it works directly with any OpenAI-compatible local server:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt54",
    base_url="http://localhost:8000/v1",
    api_key="local",        # any non-empty string for local servers
    temperature=0,
)
```

---

## Step 3: Define Tools

These four Python functions map to the same retrieval operations as the built-in file tools, implemented as LangChain `@tool` functions:

```python
import json
import glob as glob_module
import re
from langchain_core.tools import tool

DOC_PATH = "data/financebench/processing/3M_2017_10K_reconstructed.json"

@tool
def load_document(path: str) -> dict:
    """Load and return the full reconstructed document JSON."""
    with open(path) as f:
        return json.load(f)

@tool
def filter_spans(label: str = None, page_no_max: int = None, keyword: str = None) -> list:
    """
    Filter text spans from the document by label, page number, or keyword.
    label: one of 'text', 'section_header', 'table', 'list_item'
    page_no_max: return only spans on pages <= this value
    keyword: return only spans whose text contains this string (case-insensitive)
    """
    with open(DOC_PATH) as f:
        data = json.load(f)
    spans = data["texts"]
    if label:
        spans = [s for s in spans if s["label"] == label]
    if page_no_max:
        spans = [s for s in spans if s["page_no"] <= page_no_max]
    if keyword:
        spans = [s for s in spans if keyword.lower() in s["text"].lower()]
    return [{"text": s["text"], "label": s["label"], "page_no": s["page_no"],
             "path_text": s["structure"].get("path_text", "")} for s in spans]

@tool
def search_tables(keyword: str) -> list:
    """
    Search all table spans for a keyword in cell text. Returns matching table
    markdown text and their structured cells.
    Use for financial line items: revenue, net income, total assets, debt, etc.
    """
    with open(DOC_PATH) as f:
        data = json.load(f)
    results = []
    for span in data["texts"]:
        if span["label"] == "table" and "table_data" in span:
            cells = span["table_data"]["cells"]
            if any(keyword.lower() in c["text"].lower() for c in cells):
                results.append({
                    "text": span["text"],
                    "page_no": span["page_no"],
                    "cells": cells,
                })
    return results

@tool
def calculate(expression: str) -> str:
    """
    Evaluate a Python arithmetic expression and return the result as a string.
    Use for any derived metric: YoY change, margins, ratios.
    Example: '(4858 - 5050) / 5050'
    """
    result = eval(expression, {"__builtins__": {}}, {})
    return str(result)

tools = [load_document, filter_spans, search_tables, calculate]
```

---

## Step 4: Create the Agent

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a financial document QA agent.
Answer questions about SEC filings using the provided tools.

Rules:
- Always retrieve answers from the document — never answer from memory.
- For cover-page facts (name, ticker, EIN, address, shares): use filter_spans with page_no_max=2.
- For section content (segments, risk factors, legal proceedings): use filter_spans with label='section_header', then filter by path_text.
- For financial figures (revenue, net income, assets, debt): use search_tables with the line item keyword.
- For any arithmetic: use the calculate tool.
- Return a concise value or short phrase as the answer — not a full sentence."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,           # prints each tool call and result
    return_intermediate_steps=True,
)
```

---

## Step 5: Run All 30 Queries

```python
import json
import time
import os

os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

with open("data/financebench/queries.txt") as f:
    questions = [q.strip() for q in f if q.strip()]

doc_name = "3M_2017_10K.pdf"
results = {doc_name: {}}

for i, question in enumerate(questions, 1):
    start = time.time()

    response = agent_executor.invoke({"input": question})

    latency = round(time.time() - start, 2)
    steps = response.get("intermediate_steps", [])
    tools_used = list({step[0].tool for step in steps})
    num_iterations = len(steps)

    # Extract token usage if available
    usage = response.get("usage_metadata", {})
    input_tokens = usage.get("input_tokens", None)
    output_tokens = usage.get("output_tokens", None)

    results[doc_name][question] = {
        "predicted_answer": response["output"],
        "latency_seconds": latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tools_used": tools_used,
        "num_iterations": num_iterations,
    }

    # Write per-query trace log
    with open(f"logs/3M_2017_10K_{i}_trace.txt", "w") as log:
        log.write(f"=== QUERY {i} ===\n")
        log.write(f"Question: {question}\n\n")
        for j, (action, observation) in enumerate(steps, 1):
            log.write(f"--- Step {j} ---\n")
            log.write(f"Tool: {action.tool}\n")
            log.write(f"Input: {action.tool_input}\n")
            log.write(f"Retrieved: {str(observation)[:500]}\n\n")
        log.write(f"--- Final Answer ---\n")
        log.write(f"Predicted: {response['output']}\n")
        log.write(f"Latency: {latency}s | Input tokens: {input_tokens} | "
                  f"Output tokens: {output_tokens} | Iterations: {num_iterations}\n")

    print(f"[{i}/30] {question[:60]}... → {response['output'][:80]}")

with open("results/3M_2017_10K_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Done. Results saved to results/3M_2017_10K_results.json")
```

---

## Output Structure

```
project/
├── models/gpt54.py
├── agent.py                        ← the script above
├── data/financebench/
│   ├── sample.txt
│   ├── queries.txt
│   ├── sample_labels.json
│   └── processing/3M_2017_10K_reconstructed.json
├── results/
│   └── 3M_2017_10K_results.json
└── logs/
    ├── 3M_2017_10K_1_trace.txt
    ├── 3M_2017_10K_2_trace.txt
    └── ...
```

---

## Troubleshooting

**`ChatOpenAI` connection error** — Check that `models/gpt54.py` is running and the port matches `base_url`. Confirm with `curl http://localhost:8000/v1/models`.

**Tool calls not invoked** — The local model must support function calling. If it doesn't return structured tool-call JSON, `create_tool_calling_agent` will fail silently. Test with a simple function-calling request first.

**`table_data` missing on a span** — Some table spans lack parsed `table_data`. The `search_tables` tool guards against this with `"table_data" in span`. Fall back to searching the Markdown `text` field.

**Filename mapping** — Strip `.pdf`, append `_reconstructed.json`. `3M_2017_10K.pdf` → `3M_2017_10K_reconstructed.json`.
