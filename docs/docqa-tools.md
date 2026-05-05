# Document QA Tool Specifications

LangChain `@tool` functions for question answering over parsed financial documents (`*_reconstructed.json`).

The JSON format has three top-level keys — `doc_name`, `origin`, `texts` — where `texts` is an array of spans with fields: `text`, `label` (`text`/`section_header`/`table`/`list_item`), `page_no`, `bold`, `size`, `structure.level` (H1–H4/Body), `structure.path_text`, and `table_data.cells` (tables only).

---

## Tool 1: `load_document`

**LangChain custom `@tool`.** Loads the full document JSON from disk.

### When to use
- First step of every session: confirm `doc_name` and `origin.filename`.
- Inspect top-level structure before filtering.

### Specification

```python
@tool
def load_document(path: str) -> dict:
    """
    Load and return the reconstructed document JSON at the given path.
    Returns a dict with keys: doc_name, origin, texts.
    """
    with open(path) as f:
        return json.load(f)
```

### Output sample

```json
{
  "doc_name": "3M_2017_10K",
  "origin": { "filename": "3M_2017_10K.pdf", "mimetype": "application/pdf" },
  "texts": [ ... ]
}
```

---

## Tool 2: `filter_spans`

**LangChain custom `@tool`.** Primary retrieval tool — filters the `texts` array by label, page, or keyword.

### When to use
- Cover-page facts: filter `page_no_max=2`
- Section headers / segment names: filter `label="section_header"`
- Narrative text under a section: filter `label="text"` + keyword in `path_text`
- Risk factor count: filter `label="section_header"` + `path_text` contains "Item 1A"

### Specification

```python
@tool
def filter_spans(
    label: str = None,
    page_no_max: int = None,
    keyword: str = None,
    path_text_contains: str = None,
) -> list:
    """
    Filter text spans from the document JSON.

    Args:
        label: one of 'text', 'section_header', 'table', 'list_item'
        page_no_max: keep only spans on pages <= this value
        keyword: keep only spans whose text contains this string (case-insensitive)
        path_text_contains: keep only spans whose structure.path_text contains this string

    Returns:
        List of dicts with keys: text, label, page_no, path_text
    """
    with open(DOC_PATH) as f:
        spans = json.load(f)["texts"]
    if label:
        spans = [s for s in spans if s["label"] == label]
    if page_no_max:
        spans = [s for s in spans if s["page_no"] <= page_no_max]
    if keyword:
        spans = [s for s in spans if keyword.lower() in s["text"].lower()]
    if path_text_contains:
        spans = [s for s in spans
                 if path_text_contains.lower() in s["structure"].get("path_text", "").lower()]
    return [{"text": s["text"], "label": s["label"], "page_no": s["page_no"],
             "path_text": s["structure"].get("path_text", "")} for s in spans]
```

### Usage patterns

**Cover-page lookup:**
```python
filter_spans(page_no_max=2)
```

**Section headers only:**
```python
filter_spans(label="section_header")
```

**Risk factor headers:**
```python
filter_spans(label="section_header", path_text_contains="Item 1A")
```

**Keyword in narrative:**
```python
filter_spans(label="text", keyword="material weakness")
```

---

## Tool 3: `search_tables`

**LangChain custom `@tool`.** Searches structured `table_data.cells` for a keyword. Use for all financial line items.

### When to use
- Revenue, net income, total assets, long-term debt, EPS
- Trading symbols, share counts from cover-page tables
- Auditor firm name from audit report table
- Any question whose answer is a number in a financial statement

### Specification

```python
@tool
def search_tables(keyword: str) -> list:
    """
    Search all table spans for a keyword in cell text.
    Returns matching tables with their Markdown text and structured cells.

    Use for financial line items: revenue, net income, total assets, debt, EPS, etc.
    Also use for cover-page tables: trading symbols, share counts.

    Args:
        keyword: string to search for in table cell text (case-insensitive)

    Returns:
        List of dicts with keys: text (Markdown), page_no, cells (list of cell dicts)
    """
    with open(DOC_PATH) as f:
        spans = json.load(f)["texts"]
    results = []
    for span in spans:
        if span["label"] == "table" and "table_data" in span:
            cells = span["table_data"]["cells"]
            if any(keyword.lower() in c["text"].lower() for c in cells):
                results.append({
                    "text": span["text"],
                    "page_no": span["page_no"],
                    "cells": cells,
                })
    return results
```

### Output sample

```json
[
  {
    "text": "| Net Sales | 30,109 | 30,109 |\n| --- | --- | --- |\n| Operating Income | 6,217 | 5,922 |",
    "page_no": 5,
    "cells": [
      { "row": 0, "col": 0, "text": "Net Sales", "is_column_header": false, "is_row_header": true },
      { "row": 0, "col": 1, "text": "30,109", "is_column_header": false, "is_row_header": false }
    ]
  }
]
```

---

## Tool 4: `calculate`

**LangChain custom `@tool`.** Safe arithmetic execution. Always use this instead of computing in the model's head.

### When to use
- Year-over-year percentage change
- Margin calculations (gross margin, operating margin)
- Any division, multiplication, or multi-step formula

### Specification

```python
@tool
def calculate(expression: str) -> str:
    """
    Evaluate a Python arithmetic expression and return the result as a string.
    Always use this tool for any arithmetic — never compute numbers mentally.

    Args:
        expression: a valid Python arithmetic expression as a string
                    Example: '(30109 - 29140) / 29140'

    Returns:
        The numeric result as a string
    """
    result = eval(expression, {"__builtins__": {}}, {})
    return str(round(result, 6))
```

### Usage examples

```python
calculate("(30109 - 29140) / 29140")        # YoY revenue change → "0.033264"
calculate("6217 / 30109")                   # Operating margin → "0.206479"
calculate("4858 / 1000")                    # Convert millions to billions → "4.858"
```

---

## Tool Selection Guide

| Question | Primary tool | Notes |
|---|---|---|
| Registrant's exact name | `filter_spans(page_no_max=2, label="text")` | First bold span on page 1 |
| State of incorporation, IRS EIN | `filter_spans(page_no_max=2, keyword="IRS")` | Usually in cover-page text |
| Principal address, ZIP code | `filter_spans(page_no_max=2, keyword="principal")` | Cover page |
| Telephone number | `filter_spans(page_no_max=2, keyword="telephone")` | Cover page |
| Fiscal year-end | `filter_spans(page_no_max=2)` | Cover page header |
| Trading symbols and exchanges | `search_tables("Trading Symbol")` | Cover page table |
| Shares outstanding | `filter_spans(page_no_max=2, keyword="shares")` | Cover page |
| Operating segments | `filter_spans(label="section_header", path_text_contains="Item 1")` | Then read body |
| Principal products/services | `filter_spans(label="text", path_text_contains="Item 1")` | MD&A section |
| Risk factor count | `filter_spans(label="section_header", path_text_contains="Item 1A")` | Count results |
| Legal proceedings | `filter_spans(path_text_contains="Item 3")` | Yes/No disclosure |
| Liquidity discussion | `filter_spans(path_text_contains="Liquidity")` | Yes/No presence |
| Off-balance-sheet arrangements | `filter_spans(keyword="off-balance")` | Yes/No |
| Market risk disclosures | `filter_spans(path_text_contains="Item 7A")` | List risk types |
| Auditor name | `filter_spans(keyword="registered public accounting firm")` | Item 8 |
| Audit opinion type | `filter_spans(path_text_contains="Item 8", label="text")` | Read opinion paragraph |
| Total revenue | `search_tables("Net Sales")` or `search_tables("Revenue")` | Income statement |
| Net income | `search_tables("Net Income")` | Income statement |
| Total assets | `search_tables("Total Assets")` | Balance sheet |
| Long-term debt | `search_tables("Long-term debt")` | Balance sheet |
| Material weaknesses | `filter_spans(path_text_contains="Item 9A")` | Yes/No |
| Proxy statement reference | `filter_spans(keyword="proxy", label="text")` | Part III |
| Exhibit list | `filter_spans(label="list_item", path_text_contains="Exhibit")` | Exhibit index |
| Arithmetic / derived metrics | `calculate(expression)` | After retrieving raw numbers |

---

## Summary

| Tool | Type | Primary use |
|---|---|---|
| `load_document` | Custom `@tool` | Load the document JSON from disk |
| `filter_spans` | Custom `@tool` | Retrieve text/header/list spans by label, page, keyword, path |
| `search_tables` | Custom `@tool` | Search table cells for financial line items |
| `calculate` | Custom `@tool` | Safe arithmetic — YoY change, margins, ratios |

All four tools are sufficient to answer the 30 structured cover-page and financial queries without a vector index or external database.
