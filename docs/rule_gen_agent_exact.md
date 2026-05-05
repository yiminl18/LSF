# Rule Generation via Agent (Exact Answer) — `src/rule_gen_agent_exact.py`

---

## Overview

This module generates Python rules where each rule **directly returns the exact answer string** for a given question, rather than a list of retrieved spans. Unlike `rule_gen_llm_coarse` — which returns span objects that are later fed to an LLM for answer generation — rules produced here are self-contained extractors whose output is the final answer value.

A LangChain tool-calling agent iteratively writes, tests, and refines rules against ground truth answers across all sampled documents. The agent stops when it cannot improve further or a maximum iteration count is reached.

---

## Difference from `rule_gen_llm_coarse`

| Aspect | `rule_gen_llm_coarse` | `rule_gen_agent_exact` |
|---|---|---|
| Rule output type | `list[dict]` — span objects | `str` — exact answer string |
| Downstream LLM call | Yes (spans → LLM → answer) | No (rule output IS the answer) |
| Generation method | Single LLM call | Iterative agent loop with tool use |
| Verification | None at generation time | Agent tests rules against ground truth during generation |
| Goal | High recall of relevant spans | Maximum exact-match accuracy across sampled docs |
| Rule interface | `rule_<name>(doc) -> list[dict]` | `rule_<name>(doc) -> str` |

---

## Rule Interface Contract

```python
def rule_<name>(doc: dict) -> str:
    """One-line description of what this rule extracts."""
    # ... extract and return the exact answer ...
    return "answer string"   # or "" if not found
```

- Input: `doc` — a fully loaded `*_reconstructed.json` dict (same schema as all other modules)
- Output: `str` — the extracted answer, or `""` if not found
- Must never raise exceptions — return `""` on any failure
- Must be self-contained (import inside if needed)

---

## Function Interface

```python
def rule_gen_agent_exact(
    documents: list[dict],       # list of loaded document JSONs
    question: str,               # the question to answer
    ground_truth: dict,          # { "filename.pdf": "answer string", ... }
    model_name: str = "gpt54",
    output_dir: str = "results/financebench/rule_gen_exact",
    rules_dir: str = "rules_exact/financebench",
    logs_dir: str = "logs/financebench/agent",
    max_iterations: int = 10,
) -> dict:
    """
    Use a LangChain agent to iteratively generate and test Python rules
    that return the exact answer for a question across all sampled documents.

    Returns a summary dict with rule names, accuracy per rule, and run metadata.
    """
```

---

## Agent Setup

Use the same LangChain agent infrastructure as `src/agent.py`:

- **LLM:** `AzureChatOpenAI` via `src/models/gpt54.py` credentials
- **Tools:** all default LangChain tools available in the project (grep, bash/code execution, file read) — do **not** use user-defined domain tools from `src/tools/`
- **Framework:** `create_tool_calling_agent` + `AgentExecutor` (LangChain 0.3)
- **Token tracking:** capture `input_tokens`, `output_tokens` per agent step via `astream_events`
- **Trace logging:** write full agent trace to `logs/financebench/agent/`

---

## Step-by-Step Logic

### Step 1 — Prepare context for the agent

Build a structured prompt giving the agent:

1. The question and all ground truth answers keyed by `doc_name`
2. A sample of each document's `texts` array (first 80 spans)
3. Rule-writing hints (see Hint Types below)
4. The exact rule interface it must produce
5. Instructions to test each rule against all docs and report accuracy

### Step 2 — Agent iterative loop

The agent is given a system prompt instructing it to:

1. **Propose** one or more candidate rules as Python code
2. **Execute** each rule against all `N` sampled documents using a code execution tool
3. **Compare** rule output strings against ground truth (exact match, case-insensitive, strip whitespace)
4. **Report** per-doc correctness and accuracy count
5. **Refine** rules that fail on specific documents and repeat
6. **Stop** when accuracy cannot be improved further or `max_iterations` is reached

The agent loop continues until:
- All docs return the correct answer, **or**
- No improvement is made in the last iteration, **or**
- `max_iterations` agent steps are exhausted

### Step 3 — Extract final rules from agent output

Parse all Python code blocks from the agent's final response that define functions starting with `rule_`. Each such function is:
- Validated by executing it on all sample documents
- Saved as a standalone `.py` file in the rules folder
- Scored for accuracy (fraction of docs returning exact correct answer)

### Step 4 — Store results

Write each rule file and a result summary JSON.

---

## Hint Types Provided to Agent

Same 7 hint categories as `rule_gen_llm_coarse`, adapted for exact extraction:

1. **PHYSICAL LOCATION** — which page(s) is the answer on?
2. **SEMANTIC LOCATION** — which section header / `path_text` context contains it?
3. **KEYWORD PROXIMITY** — what keywords appear immediately before/after the answer?
4. **DATA FEATURE** — is the answer in a table cell? Which row/column?
5. **TYPOGRAPHY** — bold, all-caps, font size signals
6. **STRUCTURAL POSITION** — heading level, depth, sibling index
7. **TEXT PATTERN** — regex or substring pattern to extract the value from a span's text

For exact-extraction rules, the agent should prefer:
- Returning the span's `.text` field directly when it contains only the answer
- Using `re.search(pattern, span["text"]).group(1)` to extract a substring
- Reading a specific table cell value from `table_data.cells`

---

## System Prompt (Agent)

```
You are a document rule engineer. Your task is to write Python functions that
extract the exact answer to a question from a financial document JSON.

Each document has a "texts" array of span dicts with fields:
  text, label, page_no, bold, size, structure.level, structure.path_text,
  table_data.cells (when label == "table")

The rule interface is:
  def rule_<name>(doc: dict) -> str:
      """description"""
      return "exact answer string"   # or "" if not found

Your workflow:
1. Study the documents and answers I provide.
2. Write one or more candidate rule functions.
3. Test each rule by running it on all sample documents.
4. Compare outputs to ground truth (case-insensitive exact match, stripped).
5. Refine rules that fail and repeat until you cannot improve further.
6. Report the final accuracy for each rule.

Rules must:
- Never raise exceptions (return "" on any failure)
- Be self-contained (import inside the function if needed)
- Return "" when the answer is not found — never return None
```

---

## Trace Log Format

One trace file per `rule_gen_agent_exact` call, written to:

```
logs/financebench/agent/{question_slug}_{n_docs}docs_{YYYYMMDD_HHMMSS}.txt
```

Contents:
```
=== rule_gen_agent_exact ===
Question:   What is the registrant's exact name?
Slug:       what_is_the_registrants_exact_name
Docs (N):   10
Timestamp:  2026-04-28T19:00:00Z
Max iters:  10

--- Agent Turn 1 ---
[TOOL CALL] ...
[TOOL RESULT] ...
[LLM OUTPUT] ...

--- Agent Turn 2 ---
...

=== Final Rules ===
rule_exact_name_h1_page1: accuracy=10/10
rule_exact_name_bold_allcap: accuracy=9/10

=== Summary ===
Total input tokens:  4820
Total output tokens: 1105
Total latency:       38.4s
```

---

## Output File Format

### Result summary

Written to: `results/financebench/rule_gen_exact/{question_slug}_{n}_{YYYYMMDD_HHMMSS}.json`

```json
{
  "question": "What is the registrant's exact name?",
  "question_slug": "what_is_the_registrants_exact_name",
  "timestamp": "2026-04-28T19:00:00Z",
  "model": "gpt54",
  "num_documents": 10,
  "doc_names": ["AMCOR_2019_10K", "ADOBE_2022Q2_10Q", "..."],
  "max_iterations": 10,
  "agent_turns": 6,
  "latency_seconds": 38.4,
  "input_tokens": 4820,
  "output_tokens": 1105,
  "rules": [
    {
      "rule_name": "exact_name_h1_page1",
      "description": "First H1 span on page 1",
      "file": "rules_exact/financebench/what_is_the_registrants_exact_name_10/exact_name_h1_page1.py",
      "accuracy": 1.0,
      "num_correct": 10,
      "per_doc": {
        "AMCOR_2019_10K": {"predicted": "Amcor plc", "ground_truth": "Amcor plc", "correct": true}
      }
    }
  ]
}
```

### Rule file

```
rules_exact/financebench/
└── {question_slug}_{n}/
    ├── exact_name_h1_page1.py
    ├── exact_name_bold_allcap.py
    └── ...
```

Each file contains exactly one `rule_<name>(doc: dict) -> str` function.

### File naming conventions

| Artifact | Path |
|---|---|
| Result summary | `results/financebench/rule_gen_exact/{question_slug}_{n}_{YYYYMMDD_HHMMSS}.json` |
| Rule folder | `rules_exact/financebench/{question_slug}_{n}/` |
| Rule file | `rules_exact/financebench/{question_slug}_{n}/{rule_name}.py` |
| Trace log | `logs/financebench/agent/{question_slug}_{n}docs_{YYYYMMDD_HHMMSS}.txt` |

---

## Full Directory Structure

```
results/financebench/
└── rule_gen_exact/
    ├── what_is_the_registrants_exact_name_10_20260428_190000.json
    └── what_is_total_revenue_10_20260428_191500.json

rules_exact/financebench/
├── what_is_the_registrants_exact_name_10/
│   ├── exact_name_h1_page1.py
│   └── exact_name_bold_allcap.py
└── what_is_total_revenue_10/
    └── revenue_income_statement_table.py

logs/financebench/agent/
├── what_is_the_registrants_exact_name_10docs_20260428_190000.txt
└── what_is_total_revenue_10docs_20260428_191500.txt
```

---

## Edge Cases

| Situation | Behavior |
|---|---|
| Agent produces no valid rule functions | Log warning; write result with `rules: []` |
| Rule raises exception during test | Catch exception, count as incorrect, report error to agent |
| Rule returns non-string | Cast to `str`, strip; if empty or "None" treat as `""` |
| Ground truth missing for a doc | Skip that doc in accuracy calculation; note in result |
| Agent exceeds `max_iterations` | Stop loop, save best rules found so far |
| Output directory does not exist | Create all parent directories |

---

## Relation to Other Modules

| Module | Role |
|---|---|
| `src/rule_gen_llm_coarse.py` | Single-pass rule gen; rules return spans for LLM QA |
| `src/rule_gen_agent_exact.py` | This module — agent-iterative; rules return exact answer string |
| `src/rule_apply_individual.py` | Applies coarse rules (span retrieval + LLM) |
| `src/agent.py` | LangChain agent infrastructure reused here |
| `src/eval_rule.py` | Evaluates coarse rules; analogous eval for exact rules is inline |
