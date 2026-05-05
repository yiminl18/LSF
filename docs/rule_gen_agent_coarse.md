# Rule Generation via Agent (Coarse) — `src/rule_gen_agent_coarse.py`

---

## Overview

This module generates Python span-retrieval rules using an iterative LangChain agent loop. It is the **agent-driven upgrade of `rule_gen_llm_coarse`**: both produce rules whose output is `list[dict]` (a list of retrieved spans from `doc["texts"]`), and the retrieved spans are later fed to an LLM for answer generation (via `rule_apply_merge`). The key difference is that this module uses an agent that **tests, verifies, and improves rules against ground truth** during generation, rather than issuing a single prompt.

---

## Difference from `rule_gen_llm_coarse` and `rule_gen_agent_exact`

| Aspect | `rule_gen_llm_coarse` | `rule_gen_agent_coarse` | `rule_gen_agent_exact` |
|---|---|---|---|
| Rule output | `list[dict]` spans | `list[dict]` spans | `str` exact answer |
| Downstream LLM | Yes (spans → LLM → answer) | Yes (spans → LLM → answer) | No |
| Generation | Single LLM call | Iterative agent loop | Iterative agent loop |
| Verification | None at generation time | Agent tests via LLM-judged QA | Agent tests via exact string match |
| Goal | High recall | Union of rules correct on all/most docs | Rule output IS the answer |
| Rule interface | `rule_<name>(doc) -> list[dict]` | `rule_<name>(doc) -> list[dict]` | `rule_<name>(doc) -> str` |

---

## Rule Interface Contract

Identical to `rule_gen_llm_coarse`:

```python
def rule_<name>(doc: dict) -> list[dict]:
    """One-line description of what this rule matches."""
    return [span for span in doc["texts"] if ...]   # list of matching spans
```

- Input: `doc` — a fully loaded `*_reconstructed.json` dict
- Output: `list[dict]` — zero or more span dicts from `doc["texts"]`
- Must never raise exceptions — return `[]` on any failure
- Must be self-contained (import inside the function if needed)

---

## Function Interface

```python
def rule_gen_agent_coarse(
    documents: list[dict],       # list of loaded document JSONs
    question: str,               # the question to answer
    ground_truth: dict,          # { "filename.pdf": "answer string", ... }
    model_name: str = "gpt54",
    output_dir: str = "results/financebench/rule_gen_agent_coarse",
    rules_dir: str = "rules/financebench",
    logs_dir: str = "logs/financebench/agent",
    max_iterations: int = 10,
) -> dict:
    """
    Use a LangChain agent to iteratively generate, test, and refine span-retrieval
    rules. The union of all final rules should recover the correct answer in as many
    sampled documents as possible.

    Returns a summary dict matching the rule_gen_llm_coarse result format,
    with additional agent metadata fields.
    """
```

**Note:** Rules are stored in the **same** `rules/financebench/` folder as `rule_gen_llm_coarse` output, so they can be consumed by `rule_apply_merge` without changes.

---

## Agent Setup

Reuse `src/agent.py` infrastructure:

- **LLM:** `AzureChatOpenAI` via `src/models/gpt54.py` credentials
- **Tools:** all default project tools available in `src/agent.py` (grep, bash/code execution, file read) — do **not** use user-defined domain tools from `src/tools/`
- **Framework:** `create_tool_calling_agent` + `AgentExecutor` (LangChain 0.3)
- **Token tracking:** capture `input_tokens`, `output_tokens` per step via `astream_events`
- **Trace logging:** full agent trace written to `logs/financebench/agent/`

---

## Step-by-Step Logic

### Step 1 — Prepare agent context

Build a prompt giving the agent:

1. The question and all ground truth answers keyed by `doc_name`
2. First 80 spans of each document's `texts` array as JSON
3. The 7 hint types (same as `rule_gen_llm_coarse`)
4. The exact rule interface it must produce (`list[dict]`)
5. Description of the verification loop: the agent should propose rules, test them, then improve

### Step 2 — Provide the agent a `test_rules` tool

The agent is given a tool that:
1. Accepts Python code defining one or more `rule_*` functions
2. Executes each rule on all `N` sampled documents → gets `list[dict]` spans per doc
3. Concatenates span texts in reading order → `retrieved_text`
4. Calls the LLM to answer the question from `retrieved_text` (same prompt as `rule_apply_merge`)
5. Compares predicted answer to ground truth (case-insensitive, stripped)
6. Returns a JSON report:

```json
{
  "rule_name": {
    "doc_name_1": {"retrieved_text": "...", "predicted": "Amcor plc", "ground_truth": "Amcor plc", "correct": true},
    "doc_name_2": {"retrieved_text": "", "predicted": "NOT FOUND", "ground_truth": "Amcor plc", "correct": false},
    "accuracy": 0.9,
    "num_correct": 9,
    "num_spans_avg": 2.1
  }
}
```

### Step 3 — Agent iterative loop

The agent is instructed to:

1. **Propose** one or more candidate `rule_*` functions as Python code
2. **Call `test_rules`** to evaluate them against all sample documents via LLM-judged QA
3. **Analyze failures** — which docs are not answered correctly? What spans were retrieved vs. what was needed?
4. **Refine** existing rules or add new rules targeting the failing documents
5. **Check merge coverage** — does the union of current rules answer all docs correctly?
6. **Stop** when:
   - Union of rules is correct on all docs, **or**
   - No improvement in last iteration, **or**
   - `max_iterations` is reached

### Step 4 — Extract and save final rules

Parse all Python `rule_*` functions from the agent's output across all turns. For each unique rule:
- Execute on all sample docs to compute final accuracy
- Save as standalone `.py` file in `rules/financebench/{question_slug}_{n}/`
- Record per-doc coverage in result summary

### Step 5 — Compute union coverage

After saving all rules, compute the **merge accuracy**: for each doc, does at least one rule return spans that lead to the correct answer? This is the primary metric.

```python
# for each doc, union spans from all final rules → retrieved_text → LLM → compare
merge_accuracy = num_docs_correct_under_union / num_docs
```

### Step 6 — Write outputs

- Result summary JSON → `results/financebench/rule_gen_agent_coarse/{question_slug}_{n}_{timestamp}.json`
- Rule files → `rules/financebench/{question_slug}_{n}/rule_*.py`
- Trace log → `logs/financebench/agent/{question_slug}_{n}docs_{timestamp}.txt`

---

## Hint Types Provided to Agent

Same 7 categories as `rule_gen_llm_coarse`:

1. **PHYSICAL LOCATION** — which page(s) is the answer on?
2. **SEMANTIC LOCATION** — which section header / `path_text` contains it?
3. **KEYWORD PROXIMITY** — keywords adjacent to the answer span
4. **DATA FEATURE** — table label, row/column position, cell headers
5. **TYPOGRAPHY** — bold, all-caps, font size
6. **STRUCTURAL POSITION** — heading level, depth, sibling index
7. **ANY OTHER PATTERN** — be creative and exhaustive

Additionally instruct the agent: *"If no single rule covers all documents, find the minimal set of rules whose union covers all documents. Prefer fewer, more precise rules over many overlapping ones."*

---

## System Prompt (Agent)

```
You are a document rule engineer. Your task is to write Python functions that
locate the span(s) containing the answer to a question in financial document JSONs.

Each document has a "texts" array of span dicts with fields:
  text, label, page_no, bold, size, structure.level, structure.path_text,
  table_data.cells (when label == "table")

The rule interface is:
  def rule_<name>(doc: dict) -> list[dict]:
      """description"""
      return [span for span in doc["texts"] if ...]

Your workflow:
1. Study the documents and ground truth answers I provide.
2. Propose one or more candidate rule functions.
3. Use the test_rules tool to evaluate them — it runs your rules, feeds retrieved
   spans to an LLM, and compares the predicted answer to ground truth.
4. Analyze which documents fail and why (wrong spans retrieved, or none at all).
5. Refine failing rules or add new rules that cover the failing documents.
6. Check whether the UNION of all your rules now covers all documents.
7. Repeat until union accuracy is maximized or you cannot improve further.

Aim for a small, precise set of rules whose union answers every document correctly.
Never raise exceptions in rules — return [] on any failure.
```

---

## Trace Log Format

Written to: `logs/financebench/agent/{question_slug}_{n}docs_{YYYYMMDD_HHMMSS}.txt`

```
=== rule_gen_agent_coarse ===
Question:   What is the registrant's exact name?
Slug:       what_is_the_registrants_exact_name_10
Docs (N):   10
Timestamp:  2026-04-28T20:00:00Z
Max iters:  10

--- Agent Turn 1 ---
[LLM] Proposed 3 rules: rule_h1_page1, rule_bold_allcap, rule_exact_name_keyword
[TOOL CALL] test_rules(code=...)
[TOOL RESULT] rule_h1_page1: accuracy=8/10 | rule_bold_allcap: accuracy=7/10 | ...

--- Agent Turn 2 ---
[LLM] Analyzing failures on AMCOR_2019_10K and INTEL_2016_10K ...
[TOOL CALL] test_rules(code=...)  # refined rules
[TOOL RESULT] rule_h1_page1: accuracy=9/10 | rule_cover_table_bold: accuracy=6/10

--- Agent Turn 3 ---
[LLM] Union of rule_h1_page1 + rule_exact_name_keyword covers all 10 docs.
[STOP] Union accuracy = 10/10 — stopping.

=== Final Rules ===
rule_h1_page1:           accuracy=9/10  (individual)
rule_exact_name_keyword: accuracy=8/10  (individual)
Union (merge) accuracy:  10/10

=== Summary ===
Total agent turns:   3
Total input tokens:  12840
Total output tokens: 2205
Total latency:       61.3s
```

---

## Output File Format

### Result summary

Written to: `results/financebench/rule_gen_agent_coarse/{question_slug}_{n}_{YYYYMMDD_HHMMSS}.json`

Same schema as `rule_gen_llm_coarse` result, with additional fields:

```json
{
  "question": "What is the registrant's exact name?",
  "question_slug": "what_is_the_registrants_exact_name",
  "timestamp": "2026-04-28T20:00:00Z",
  "model": "gpt54",
  "num_documents": 10,
  "doc_names": ["AMCOR_2019_10K", "ADOBE_2022Q2_10Q", "..."],
  "max_iterations": 10,
  "agent_turns": 3,
  "latency_seconds": 61.3,
  "input_tokens": 12840,
  "output_tokens": 2205,
  "merge_accuracy": 1.0,
  "merge_num_correct": 10,
  "rules": [
    {
      "rule_name": "rule_h1_page1",
      "description": "First H1 span on page 1",
      "file": "rules/financebench/what_is_the_registrants_exact_name_10/rule_h1_page1.py",
      "individual_accuracy": 0.9,
      "individual_num_correct": 9
    },
    {
      "rule_name": "rule_exact_name_keyword",
      "description": "Span adjacent to 'exact name of registrant' keyword",
      "file": "rules/financebench/what_is_the_registrants_exact_name_10/rule_exact_name_keyword.py",
      "individual_accuracy": 0.8,
      "individual_num_correct": 8
    }
  ],
  "merge_per_doc": {
    "AMCOR_2019_10K": {"predicted": "Amcor plc", "ground_truth": "Amcor plc", "correct": true},
    "INTEL_2016_10K": {"predicted": "Intel Corporation", "ground_truth": "Intel Corporation", "correct": true}
  }
}
```

Additional fields vs `rule_gen_llm_coarse`:

| Field | Type | Description |
|---|---|---|
| `agent_turns` | int | Number of agent iterations run |
| `merge_accuracy` | float | Fraction of docs correct under union of all rules |
| `merge_num_correct` | int | Docs correctly answered by union |
| `rules[].individual_accuracy` | float | Fraction correct for this rule alone |
| `rules[].individual_num_correct` | int | Docs correct for this rule alone |
| `merge_per_doc` | dict | Per-doc result under union of all rules |

### Rule files

Stored in the **same** location as `rule_gen_llm_coarse` output:

```
rules/financebench/
└── {question_slug}_{n}/
    ├── rule_h1_page1.py
    ├── rule_exact_name_keyword.py
    └── ...
```

Rules generated by this module coexist with rules from `rule_gen_llm_coarse` in the same folder and can be consumed by `rule_apply_merge` without modification.

---

## Full Directory Structure

```
results/financebench/
└── rule_gen_agent_coarse/
    ├── what_is_the_registrants_exact_name_10_20260428_200000.json
    └── what_is_total_revenue_10_20260428_201500.json

rules/financebench/
└── what_is_the_registrants_exact_name_10/
    ├── rule_h1_page1.py                  ← from agent_coarse
    ├── rule_exact_name_keyword.py        ← from agent_coarse
    ├── rule_cover_page_bold_header.py    ← from rule_gen_llm_coarse (coexists)
    └── ...

logs/financebench/agent/
├── what_is_the_registrants_exact_name_10docs_20260428_200000.txt
└── what_is_total_revenue_10docs_20260428_201500.txt
```

---

## Edge Cases

| Situation | Behavior |
|---|---|
| Agent produces no valid rule functions | Log warning; write result with `rules: []`, `merge_accuracy: 0.0` |
| Rule raises exception during test | Catch it, count as incorrect for all docs, report error text to agent |
| `test_rules` LLM call fails | Return error string to agent; do not crash |
| Agent reaches `max_iterations` without full coverage | Save best rules found; record `agent_turns = max_iterations` |
| Rule file already exists in rules folder | Overwrite — agent-generated rules supersede prior versions with same name |
| Ground truth missing for a doc | Skip that doc in accuracy; note it in result |

---

## Relation to Other Modules

| Module | Role |
|---|---|
| `src/rule_gen_llm_coarse.py` | Single-pass coarse rule gen — no verification |
| `src/rule_gen_agent_coarse.py` | This module — agent-iterative coarse rule gen with LLM-judged verification |
| `src/rule_gen_agent_exact.py` | Agent-iterative; rules return exact answer string (no downstream LLM) |
| `src/rule_apply_merge.py` | Consumes rules from this module unchanged |
| `src/agent.py` | LangChain agent infrastructure reused here |
| `src/eval_rule.py` | Post-hoc evaluation of rules on held-out docs |
