# Rule Apply — Individual Strategy — `src/rule_apply_individual.py`

---

## Overview

This module applies a single named rule to a single document to retrieve candidate spans, converts those spans to text in reading order, and calls an LLM to generate an answer for a given question. Results are stored in a structured folder under `data/financebench/rule_run/`.

The "individual" strategy means: one rule is applied independently per document — no cross-document aggregation or rule combination at retrieval time.

---

## Function Interface

```python
def rule_apply_individual(
    document: dict,              # fully loaded *_reconstructed.json dict
    rule_name: str,              # e.g. "cover_page_bold_header"
    question_slug: str,          # slug identifying the question (matches rule folder)
    question: str,               # full question text
    model_name: str = "gpt54",  # model under src/models/
    rules_dir: str = "rules/financebench",
    output_dir: str = "results/financebench/rule_run/individual",
) -> dict:
    """
    Apply a named rule to a document, retrieve matching spans, and call the LLM
    to answer the question from the retrieved text.

    Returns a result dict for this (rule, document, question) triple.
    """
```

---

## Step-by-Step Logic

### Step 1 — Load and apply the rule

Load the rule Python file from:
```
rules/financebench/{question_slug}/{rule_name}.py
```

Call the rule function:
```python
matching_spans = rule_fn(document)   # list[dict] — matching spans from doc["texts"]
```

### Step 2 — Convert spans to text in reading order

Sort matching spans by `(page_no, structure.level_index)` to preserve reading order. Then concatenate their `text` fields with newline separators:

```python
sorted_spans = sorted(matching_spans, key=lambda s: (
    s["page_no"],
    s["structure"].get("level_index", 0)
))
retrieved_text = "\n\n".join(s["text"] for s in sorted_spans)
```

If no spans are returned, `retrieved_text = ""` and the LLM is still called but with an empty context (so the model can signal "not found").

### Step 3 — Count retrieved tokens

Count the tokens in `retrieved_text` before calling the LLM. This is recorded as `retrieved_token_count` — a measure of how much context the rule extracted.

### Step 4 — Call LLM to generate answer

Build the prompt:

**System:**
```
You are a financial document QA assistant.
You are given a passage extracted from a financial filing and a question.
Answer the question using only the provided passage.
If the passage does not contain enough information to answer, reply with "NOT FOUND".
Return only the answer — a short value or phrase, not a full sentence.
```

**User:**
```
Passage:
{retrieved_text}

Question: {question}
```

Record `input_tokens`, `output_tokens`, `latency_seconds` from the LLM response.

### Step 5 — Store result and merge into output file

Each call produces one result record:

```json
{
  "rule_name": "cover_page_bold_header",
  "question_slug": "what_is_the_registrants_exact_name",
  "question": "What is the registrant's exact name?",
  "doc_name": "3M_2017_10K",
  "strategy": "individual",
  "predicted_answer": "3M Company",
  "latency_seconds": 2.1,
  "input_tokens": 312,
  "output_tokens": 8,
  "retrieved_token_count": 245,
  "retrieved_spans": [
    {
      "text": "3M COMPANY",
      "label": "section_header",
      "page_no": 1,
      "bold": 1,
      "structure": { "level": "H1", "path_text": "" }
    }
  ],
  "retrieved_text": "3M COMPANY"
}
```

This record is appended to the merged output file for this (rule, question) pair.

---

## Output File Format

### Merged result file

All results for a given (rule, question) pair — across all documents run — are merged into one JSON file:

```
results/financebench/rule_run/
└── individual/
    └── {question_slug}/
        └── {rule_name}_individual.json
```

Example path:
```
results/financebench/rule_run/individual/what_is_the_registrants_exact_name_10/cover_page_bold_header_individual.json
```

The file is a JSON array. Each element is one result record (one document):

```json
[
  {
    "rule_name": "cover_page_bold_header",
    "question_slug": "what_is_the_registrants_exact_name",
    "question": "What is the registrant's exact name?",
    "doc_name": "3M_2017_10K",
    "strategy": "individual",
    "predicted_answer": "3M Company",
    "latency_seconds": 2.1,
    "input_tokens": 312,
    "output_tokens": 8,
    "retrieved_token_count": 245,
    "retrieved_spans": [ ... ],
    "retrieved_text": "3M COMPANY"
  },
  {
    "doc_name": "ADOBE_2022Q2_10Q",
    "predicted_answer": "Adobe Inc.",
    ...
  }
]
```

When called for a new document, the function **reads the existing file** (if present), appends the new record, and writes it back. This allows incremental runs across documents.

### Field reference

| Field | Type | Description |
|---|---|---|
| `rule_name` | string | Name of the applied rule |
| `question_slug` | string | Slug of the question |
| `question` | string | Full question text |
| `doc_name` | string | `doc["doc_name"]` from the document JSON |
| `strategy` | string | Always `"individual"` for this module |
| `predicted_answer` | string | LLM-generated answer |
| `latency_seconds` | float | Wall-clock time for LLM call |
| `input_tokens` | int | Tokens sent to the LLM (system + user prompt) |
| `output_tokens` | int | Tokens in LLM response |
| `retrieved_token_count` | int | Token count of `retrieved_text` before LLM call |
| `retrieved_spans` | list[dict] | Matching spans returned by the rule |
| `retrieved_text` | string | Concatenated text from spans in reading order |

---

## Full Directory Structure

```
results/financebench/
└── rule_run/
    └── individual/
        ├── what_is_the_registrants_exact_name_10/
        │   ├── cover_page_bold_header_individual.json
        │   ├── registrant_keyword_proximity_individual.json
        │   └── h1_span_page1_individual.json
        ├── what_are_the_trading_symbols_and_listing_exchanges_10/
        │   ├── cover_page_table_trading_symbol_individual.json
        │   └── ticker_keyword_table_cell_individual.json
        └── ...
```

---

## Reading Order Sorting

Spans are sorted before concatenation using:

```python
key = (page_no, structure.level_index)
```

`level_index` is the position of the span among its siblings within its parent section, preserving the original document order within a page. If `level_index` is absent, fall back to the span's position in the original `texts` array.

---

## Token Counting

Use the same tokenizer as the LLM backend (or approximate with `len(text.split()) * 1.3` if no tokenizer is available). `retrieved_token_count` is measured on `retrieved_text` only — it represents the cost of the rule's retrieval, independent of the system/question prompt overhead.

---

## Edge Cases

| Situation | Behavior |
|---|---|
| Rule returns empty list | `retrieved_text = ""`, `retrieved_token_count = 0`, LLM called with empty passage, expected to return `"NOT FOUND"` |
| Rule file not found | Raise `FileNotFoundError` with a clear message including the expected path |
| LLM returns no content | `predicted_answer = null` in result |
| Output file already exists | Read existing array, append new record, write back |
| Output directory does not exist | Create it (including all parent directories) |

---

## Relation to Other Modules

| Module | Role |
|---|---|
| `src/rule_gen_llm_coarse.py` | Generates the rule `.py` files under `rules/financebench/` |
| `src/rule_apply_individual.py` | This module — applies one rule to one document |
| `src/run_financebench.py` | Baseline agent that answers questions without rules |
