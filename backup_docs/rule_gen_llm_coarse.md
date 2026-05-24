# Rule Generation via LLM (Coarse) — `src/rule_gen_llm_coarse.py`

---

## Problem Setting

A collection of documents share strong structural similarity — they are all generated from the same template (e.g., SEC 10-K filings, 10-Q filings). Given a fixed question, the location of the answer across all documents follows consistent patterns. For example, "What is the registrant's exact name?" is always found on page 1 in a bold header, and "What is total revenue?" is always in the income statement table.

The goal of this module is to use an LLM to **discover and codify these patterns as executable rules**. Each rule is a Python function that, given any document from the collection, returns the list of JSON nodes whose content contains the answer.

---

## Document JSON Format

Each document is a `*_reconstructed.json` file with the following structure (see `data/financebench/processing/3M_2023Q2_10Q_reconstructed.json`):

```json
{
  "doc_name": "3M_2023Q2_10Q",
  "origin": {
    "mimetype": "application/pdf",
    "binary_hash": 14862264175189151635,
    "filename": "3M_2023Q2_10Q.pdf"
  },
  "texts": [
    {
      "text": "UNITED STATES",
      "label": "section_header",
      "page_no": 1,
      "bold": 1,
      "size": 13.0,
      "font": "TimesNewRomanPS-BoldMT",
      "all_cap": 1,
      "num_st": 0,
      "is_center": 0,
      "is_underline": 0,
      "structure": {
        "level": "H1",
        "level_index": 1,
        "parent_id": null,
        "path_text": "",
        "depth": 1,
        "h1_index_norm": 0,
        "sibling_index_norm": 0.0,
        "is_first_child": true,
        "is_last_child": false
      },
      "table_data": {
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
    }
  ]
}
```

Key fields per span:

| Field | Type | Values |
|---|---|---|
| `text` | string | Content; tables in Markdown pipe format |
| `label` | string | `"text"`, `"section_header"`, `"table"`, `"list_item"` |
| `page_no` | int | Page number in original PDF |
| `bold`, `size` | int/float | Typography |
| `structure.level` | string | `"H1"`, `"H2"`, `"H3"`, `"H4"`, `"Body"` |
| `structure.path_text` | string | Ancestor breadcrumb, e.g. `"3M COMPANY \| Item 1"` |
| `structure.depth` | int | Nesting depth |
| `table_data` | object | Only present when `label == "table"` |
| `table_data.cells` | list | Structured cell array with row/col/text |

---

## Function Interface

**File:** `src/rule_gen_llm_coarse.py`

```python
def rule_gen_llm_coarse(
    documents: list[dict],       # list of loaded document JSONs
    question: str,               # the question whose answer location to generalize
    ground_truth: dict,          # { "filename.pdf": "answer string", ... }
    model_name: str = "gpt54",   # model under src/models/
    output_dir: str = "results/financebench/rule_gen",
    rules_dir: str = "rules/financebench",
) -> dict:
    """
    Given a collection of similar documents, a question, and ground truth answers,
    use an LLM to generate Python rules that locate the answer span in any document.

    Returns a summary dict with rule names, file paths, and run metadata.
    """
```

---

## LLM Prompt Design

The prompt instructs the LLM to act as a rule engineer and generate as many rules as possible, covering all observable patterns in the document collection.

### System prompt

```
You are a document rule engineer. Your task is to analyze how a specific question
is answered across a collection of structurally similar documents, and generate
Python rules that can reliably locate the answer in any new document from the
same collection.

Each document is represented as a JSON object with a "texts" array. Each element
in "texts" is a span with these fields:
  - text: the content (tables are in Markdown pipe format)
  - label: one of "text", "section_header", "table", "list_item"
  - page_no: integer page number
  - bold: 1 if bold, 0 otherwise
  - size: font size in points
  - structure.level: heading level — "H1", "H2", "H3", "H4", or "Body"
  - structure.path_text: breadcrumb of ancestor section headers
  - table_data.cells: list of {row, col, text, is_column_header, is_row_header}
    (only present when label == "table")
```

### User prompt template

```
I have a collection of {N} financial documents that are structurally similar
(all generated from the same SEC filing template). I want to find rules that
describe WHERE the answer to the following question is located across all documents.

QUESTION: {question}

Here are the documents with their ground truth answers:

{for each document:}
--- Document: {doc_name} ---
Answer: {ground_truth[doc_name]}
Document JSON (texts array, first 80 spans shown):
{json.dumps(doc["texts"][:80], indent=2)}

---

Based on the documents and answers above, generate as many Python rules as possible
that locate the answer span(s) in any new document from this collection.

For each rule, consider the following signal types as hints:

1. PHYSICAL LOCATION — Which page(s) does the answer consistently appear on?
   Example: "answer is always on page 1 or 2"

2. SEMANTIC LOCATION — Which section header is the answer under?
   Use structure.path_text or nearby section_header spans.
   Example: "answer is under a span whose path_text contains 'Item 1'"

3. KEYWORD PROXIMITY — What keywords appear near the answer?
   Example: "answer is in a span whose text contains 'Employer Identification'"

4. DATA FEATURE — Is the answer in a table, and if so, what is the table about?
   Use label == "table" and table_data.cells.
   Example: "answer is in a table cell in row where col 0 text == 'Net Sales'"

5. TYPOGRAPHY — Is the answer in a bold span, large font, or all-caps heading?
   Example: "answer is in the first bold, all_cap span on page 1"

6. STRUCTURAL POSITION — What is the heading level or depth of the span?
   Example: "answer is in an H1 span near the top of the document"

7. ANY OTHER RULE TYPE you observe that is not listed above. Think carefully
   about patterns in the data — label combinations, sibling relationships,
   page ranges, table column/row header patterns, list item positions, etc.
   Be creative and exhaustive.

REQUIREMENTS FOR EACH RULE:
- Give the rule a short descriptive name (snake_case)
- Write it as a Python function with this exact signature:
    def rule_<name>(doc: dict) -> list[dict]:
        """One-line description of what this rule matches."""
        ...
        return [span, ...]   # list of matching spans from doc["texts"]
- The function must be self-contained (import json/re inside if needed)
- Return an empty list if no match is found — never raise exceptions
- Aim for high recall: prefer returning a few extra spans over missing the answer
- Generate as many rules as possible — cover every pattern you observe
- Rules may overlap; that is fine
```

---

## Output Format

### Result file

Written to: `results/financebench/rule_gen/{question_slug}_{timestamp}.json`

`question_slug` is the question lowercased, punctuation stripped, spaces replaced with `_`, truncated to 60 chars.

```json
{
  "question": "What is the registrant's exact name?",
  "question_slug": "what_is_the_registrants_exact_name",
  "timestamp": "2025-04-27T14:32:11Z",
  "model": "gpt54",
  "num_documents": 10,
  "doc_names": [
    "ADOBE_2022Q2_10Q",
    "AMAZON_2016_10K",
    "AMAZON_2018_10K"
  ],
  "latency_seconds": 8.4,
  "input_tokens": 18420,
  "output_tokens": 3105,
  "rules": [
    {
      "rule_name": "cover_page_bold_header",
      "description": "First bold all-caps span on page 1",
      "file": "rules/financebench/what_is_the_registrants_exact_name_10/cover_page_bold_header.py"
    },
    {
      "rule_name": "registrant_keyword_proximity",
      "description": "Span containing 'registrant' on page 1",
      "file": "rules/financebench/what_is_the_registrants_exact_name_10/registrant_keyword_proximity.py"
    }
  ]
}
```

### Rules folder

Each rule is stored as a standalone Python file. The folder name includes the number of documents used to generate the rules (`_{n}`):

```
rules/financebench/
└── {question_slug}_{n}/
    ├── cover_page_bold_header.py
    ├── registrant_keyword_proximity.py
    ├── h1_span_page1.py
    └── ...
```

### Rule file format

Each `.py` file contains exactly one function with the uniform interface:

```python
def rule_cover_page_bold_header(doc: dict) -> list[dict]:
    """Return the first bold all-caps span on page 1 — likely the registrant name."""
    return [
        span for span in doc["texts"]
        if span.get("page_no", 0) <= 1
        and span.get("bold", 0) == 1
        and span.get("all_cap", 0) == 1
    ]
```

**Uniform interface contract:**
- Input: `doc` — a fully loaded `*_reconstructed.json` dict
- Output: `list[dict]` — zero or more span dicts from `doc["texts"]` whose `text` field contains or is adjacent to the answer
- Must never raise exceptions — return `[]` on any failure

---

## File Naming Conventions

| Artifact | Path pattern |
|---|---|
| Result summary | `results/financebench/rule_gen/{question_slug}_{YYYYMMDD_HHMMSS}.json` |
| Rule folder | `rules/financebench/{question_slug}_{n}/` |
| Rule file | `rules/financebench/{question_slug}_{n}/{rule_name}.py` |

`question_slug` derivation:
```python
import re
slug = question.lower()
slug = re.sub(r"[^\w\s]", "", slug)   # strip punctuation
slug = re.sub(r"\s+", "_", slug)       # spaces to underscores
slug = slug[:60]                        # truncate
```

---

## Metadata Recorded

| Field | Source |
|---|---|
| `latency_seconds` | wall-clock time from LLM call start to response complete |
| `input_tokens` | from LLM response usage metadata |
| `output_tokens` | from LLM response usage metadata |
| `num_documents` | length of input `documents` list |
| `doc_names` | list of `doc_name` values from each input document |
| `model` | model name passed to function |
| `rules[].rule_name` | parsed from LLM output |
| `rules[].description` | docstring of generated function |
| `rules[].file` | path where rule `.py` was written |

---

## Directory Structure After Running

```
results/financebench/
└── rule_gen/
    ├── what_is_the_registrants_exact_name_20250427_143211.json
    ├── what_are_the_trading_symbols_20250427_143305.json
    └── ...

rules/financebench/
├── what_is_the_registrants_exact_name_10/
│   ├── cover_page_bold_header.py
│   ├── registrant_keyword_proximity.py
│   └── h1_span_page1.py
├── what_are_the_trading_symbols_10/
│   ├── cover_page_table_trading_symbol.py
│   └── ticker_keyword_table_cell.py
└── ...
```

---

## Applying a Rule

To apply a rule to a new document:

```python
import importlib.util, json

def load_rule(rule_file: str):
    spec = importlib.util.spec_from_file_location("rule", rule_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # return the first function whose name starts with "rule_"
    return next(v for k, v in vars(mod).items() if k.startswith("rule_"))

doc = json.load(open("data/financebench/processing/3M_2023Q2_10Q_reconstructed.json"))
rule_fn = load_rule("rules/financebench/what_is_the_registrants_exact_name_10/cover_page_bold_header.py")
matching_spans = rule_fn(doc)
for span in matching_spans:
    print(span["text"])
```

---

## Notes

- The LLM is instructed to generate rules that together achieve **high recall** — the union of all rules for a question should recover the answer in every document.
- Individual rules may have lower precision (returning extra spans); a downstream filtering or ranking step can select the best span.
- Rules are intentionally coarse at this stage — a separate refinement pass (`rule_gen_llm_fine`) can prune or specialize them using held-out documents.
- If the LLM output cannot be parsed into valid Python, the raw text is saved alongside the result JSON for manual inspection.
