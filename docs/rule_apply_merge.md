# Rule Apply — Merge Strategy — `src/rule_apply_merge.py`

---

## Overview

This module is identical to `rule_apply_individual.py` with one difference: instead of taking a single rule as input, it takes a **set of rules**, applies each independently, and takes the **union** of all retrieved spans before calling the LLM. This gives broader coverage — if any one rule locates the answer, it will be included in the context.

---

## Difference from `rule_apply_individual`

| Aspect | `rule_apply_individual` | `rule_apply_merge` |
|---|---|---|
| Rules input | single `rule_name: str` | `rule_names: list[str]` |
| Retrieval | one rule applied | all rules applied, spans unioned |
| Deduplication | n/a | spans deduplicated by identity before sort |
| Strategy tag | `"individual"` | `"merge"` |
| Output filename | `{rule_name}_individual.json` | `{rule_set_slug}_merge.json` |
| Everything else | — | identical |

---

## Function Interface

```python
def rule_apply_merge(
    document: dict,              # fully loaded *_reconstructed.json dict
    rule_names: list[str],       # e.g. ["rule_exact_name_parent_h1", "rule_page1_first_h1"]
    question_slug: str,          # slug identifying the question (matches rule folder)
    question: str,               # full question text
    model_name: str = "gpt54",  # model under src/models/
    rules_dir: str = "rules/financebench",
    output_dir: str = "results/financebench/rule_run/merge",
) -> dict:
    """
    Apply a set of rules to a document, union the retrieved spans,
    and call the LLM to answer the question from the merged retrieved text.

    Returns a result dict for this (rule_set, document, question) triple.
    """
```

---

## Step-by-Step Logic

### Step 1 — Apply each rule and union spans

For each rule in `rule_names`:
- Load `{rules_dir}/{question_slug}/{rule_name}.py` dynamically
- Call `rule_fn(document)` → get `matching_spans: list[dict]`
- Track which rules returned spans (for logging)

Union all spans across rules, **deduplicated by object identity** (a span is uniquely identified by its position in the original `texts` array — use its index):

```python
seen_indices = set()
union_spans = []
for span in all_retrieved:
    idx = document["texts"].index(span)   # position in original array
    if idx not in seen_indices:
        seen_indices.add(idx)
        union_spans.append(span)
```

If a rule file is missing, skip it and log a warning — do not raise an exception.

### Step 2 — Sort union spans in reading order

Same as `rule_apply_individual`:

```python
sorted_spans = sorted(union_spans, key=lambda s: (
    s["page_no"],
    s["structure"].get("level_index", 0)
))
retrieved_text = "\n\n".join(s["text"] for s in sorted_spans)
```

### Step 3 — Count retrieved tokens

Count tokens in `retrieved_text` → `retrieved_token_count`. Same method as `rule_apply_individual` (tiktoken or word-count approximation).

### Step 4 — Call LLM to generate answer

Identical prompt to `rule_apply_individual`:

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

Record `input_tokens`, `output_tokens`, `latency_seconds`.

### Step 5 — Store result

Same schema as `rule_apply_individual`, with two additions:

```json
{
  "rule_names": ["rule_exact_name_parent_h1", "rule_page1_first_h1", "rule_cover_page_bold_header"],
  "rule_set_slug": "rule_exact_name_parent_h1__rule_page1_first_h1__rule_cover_page_bold_header",
  "question_slug": "what_is_the_registrants_exact_name_10",
  "question": "What is the registrant's exact name?",
  "doc_name": "AMCOR_2019_10K",
  "strategy": "merge",
  "predicted_answer": "Amcor plc",
  "latency_seconds": 2.3,
  "input_tokens": 580,
  "output_tokens": 6,
  "retrieved_token_count": 510,
  "rules_with_hits": ["rule_exact_name_parent_h1", "rule_cover_page_bold_header"],
  "rules_with_no_hits": ["rule_page1_first_h1"],
  "num_spans_before_dedup": 5,
  "num_spans_after_dedup": 3,
  "retrieved_spans": [ ... ],
  "retrieved_text": "Amcor plc\n\nAmcor plc"
}
```

Additional fields vs `rule_apply_individual`:

| Field | Type | Description |
|---|---|---|
| `rule_names` | list[string] | All rule names passed as input |
| `rule_set_slug` | string | Rules joined by `__`, used in output filename |
| `rules_with_hits` | list[string] | Rules that returned at least one span |
| `rules_with_no_hits` | list[string] | Rules that returned empty list |
| `num_spans_before_dedup` | int | Total spans across all rules before deduplication |
| `num_spans_after_dedup` | int | Unique spans after deduplication |

---

## Output File Format

Same as `rule_apply_individual` — a JSON array, one record per document, appended incrementally.

### Path

```
{output_dir}/{question_slug}/{rule_set_slug}_merge.json
```

### `rule_set_slug` derivation

Join sorted rule names with `__`, truncate to 120 chars:

```python
rule_set_slug = "__".join(sorted(rule_names))[:120]
```

Sorting ensures the same set of rules always maps to the same filename regardless of input order.

---

## Full Directory Structure

```
results/financebench/
└── rule_run/
    └── merge/
        └── {question_slug}/
            └── {rule_set_slug}_merge.json
```

Example:
```
results/financebench/rule_run/merge/
└── what_is_the_registrants_exact_name_10/
    └── rule_cover_page_bold_header__rule_exact_name_parent_h1__rule_page1_first_h1_merge.json
```

---

## Edge Cases

| Situation | Behavior |
|---|---|
| All rules return empty | `retrieved_text = ""`, `retrieved_token_count = 0`, LLM called, expected `"NOT FOUND"` |
| One rule file missing | Skip it, add to `rules_with_no_hits`, log warning — do not raise |
| All rule files missing | Raise `FileNotFoundError` listing all missing paths |
| Duplicate spans across rules | Deduplicated by index in `texts` array — each span appears once |
| Output file already exists | Read existing array, append new record, write back |

---

## Relation to Other Modules

| Module | Role |
|---|---|
| `src/rule_gen_llm_coarse.py` | Generates the rule `.py` files consumed here |
| `src/rule_apply_individual.py` | Applies one rule per call — baseline comparison |
| `src/rule_apply_merge.py` | This module — unions spans from multiple rules |
| `src/eval_rule.py` | Evaluates accuracy of predictions from either strategy |
