# Rule Application

This document describes the two rule application strategies in the LSF codebase. Both take a loaded document and one or more span-retrieval rules, retrieve matching spans, and call an LLM to answer a question from the retrieved text.

---

## Strategy comparison

| Aspect | Individual | Merge |
|--------|-----------|-------|
| Code | `src/rule_apply_individual.py` | `src/rule_apply_merge.py` |
| Rules input | single `rule_name: str` | `rule_names: list[str]` |
| Retrieval | one rule applied | all rules applied, spans unioned + deduplicated |
| Strategy tag | `"individual"` | `"merge"` |
| Output filename | `{rule_name}_individual.json` | `{rule_set_slug}_merge.json` |
| Use case | Evaluate a single rule in isolation | Apply a selected rule subset at inference |

---

## Strategy 1 — Individual

**Code:** `src/rule_apply_individual.py`

### Description

Applies a single named rule to a single document. Retrieves matching spans, concatenates them in reading order, and calls the LLM to answer the question. Used to evaluate rules in isolation and to measure per-rule coverage.

### Interface

```python
def rule_apply_individual(
    document: dict,
    rule_name: str,
    question_slug: str,
    question: str,
    model_name: str = "gpt54",
    rules_dir: str = "rules/financebench",
    output_dir: str = "results/financebench/rule_run/individual",
) -> dict
```

### Logic

1. Load `{rules_dir}/{question_slug}/{rule_name}.py` and call `rule_fn(document)` → `list[dict]` spans.
2. Sort spans by `(page_no, structure.level_index)` → reading order.
3. Concatenate `span["text"]` with `\n\n` separators → `retrieved_text`.
4. Call LLM: system prompt instructs answer-only from passage, `"NOT FOUND"` if insufficient.
5. Append result record to `{output_dir}/{question_slug}/{rule_name}_individual.json`.

### Output schema

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
  "retrieved_spans": [...],
  "retrieved_text": "3M COMPANY"
}
```

Output file is a JSON array appended incrementally (one record per document).

---

## Strategy 2 — Merge

**Code:** `src/rule_apply_merge.py`

### Description

Applies a set of rules to a single document, takes the union of all retrieved spans (deduplicated by position in `doc["texts"]`), and calls the LLM once on the merged context. This is the standard inference-time strategy — rule selection algorithms (Pareto, agentic) produce the rule subset `S` that this module applies.

### Interface

```python
def rule_apply_merge(
    document: dict,
    rule_names: list[str],
    question_slug: str,
    question: str,
    model_name: str = "gpt54",
    rules_dir: str = "rules/financebench",
    output_dir: str = "results/financebench/rule_run/merge",
) -> dict
```

### Logic

1. For each rule in `rule_names`: load and apply → collect spans. Skip missing rule files with a warning.
2. Deduplicate spans by index in `doc["texts"]`:
   ```python
   seen = set()
   union_spans = [s for s in all_spans if (i := doc["texts"].index(s)) not in seen and not seen.add(i)]
   ```
3. Sort union spans by `(page_no, structure.level_index)` → reading order.
4. Call LLM on merged `retrieved_text` (same prompt as Individual).
5. Append result to `{output_dir}/{question_slug}/{rule_set_slug}_merge.json`.

`rule_set_slug` = `"__".join(sorted(rule_names))[:120]` — stable across input orderings.

### Output schema

Same fields as Individual, plus:

```json
{
  "rule_names": ["rule_exact_name_parent_h1", "rule_page1_first_h1"],
  "rule_set_slug": "rule_exact_name_parent_h1__rule_page1_first_h1",
  "strategy": "merge",
  "rules_with_hits": ["rule_exact_name_parent_h1"],
  "rules_with_no_hits": ["rule_page1_first_h1"],
  "num_spans_before_dedup": 5,
  "num_spans_after_dedup": 3,
  ...
}
```

---

## Shared conventions

**LLM prompt (both strategies):**

```
System: You are a financial document QA assistant. Answer using only the provided
        passage. Reply "NOT FOUND" if insufficient. Return only the answer value.

User:   Passage: {retrieved_text}
        Question: {question}
```

**Reading order sort:** `(page_no, structure.level_index)` — preserves original document order within a page. Falls back to position in `texts` array if `level_index` absent.

**Output files:** JSON arrays appended incrementally. Each call reads the existing file, appends one record, and writes back — safe for incremental runs across documents.

**Edge cases:**

| Situation | Behavior |
|-----------|----------|
| Rule returns no spans | `retrieved_text = ""`, LLM called, expected `"NOT FOUND"` |
| Rule file missing (merge) | Skip + warning; raise only if all files missing |
| Rule file missing (individual) | Raise `FileNotFoundError` with expected path |
| Output file exists | Read, append, write back |

---

## Output directory layout

```
results/financebench/rule_run/
├── individual/
│   └── {question_slug}/
│       └── {rule_name}_individual.json
└── merge/
    └── {question_slug}/
        └── {rule_set_slug}_merge.json
```
