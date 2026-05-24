# Rule Generation via Agent (Best Setup) — `src/rule_gen_agent_langchain.py`

---

## Overview

This is the recommended agent-based rule generation strategy. It improves over `rule_gen_agent_coarse_claude.py` in four key ways:

1. **Coverage-first rule design**: each rule is written to cover as many sampled documents as possible — broad rules first, targeted rules only for remaining uncovered docs
2. **Explicit merge accuracy target**: the primary objective is merge accuracy ≥ 0.95, where merge accuracy is the LLM accuracy on the union of text retrieved by all rules across each document
3. **Fast feedback loop**: per-rule testing uses substring match (no LLM), reserving LLM judge calls only for final union verification (`test_union`)
4. **Diagnosis-driven refinement**: `diagnose_failing_doc` shows a structural diff between failing and passing docs, giving the agent targeted evidence to fix specific failures — not just a score

---

## Comparison to Existing Modules

| Aspect | `rule_gen_llm_coarse` | `rule_gen_agent_coarse_claude` | `rule_gen_agent` (this) |
|---|---|---|---|
| Document analysis | First 80 spans (raw) | `inspect_doc` (raw spans) | `summarize_answer_locations` (pattern table) + `inspect_answer_context` (answer-anchored) |
| Rule validation | Prompt only | Prompt only | Tool-level (`write_rule` rejects bad code) |
| Per-rule feedback | None | LLM judge (expensive) | Substring match (free) |
| Final verification | None | LLM judge | LLM judge (`test_union`) |
| Failure diagnosis | None | Retrieved text shown | Structural diff vs. passing docs |
| LLM calls per run | 1 | O(rules × iters) | O(iters) — typically 3–5 `test_union` calls |

---

## Function Interface

```python
def rule_gen_agent(
    documents: list[dict],       # loaded *_reconstructed.json dicts
    question: str,               # full question text
    ground_truth: dict,          # { "DOCNAME.pdf": "answer string" }
    model_name: str = "gpt54",
    rules_dir: str = "rules/llm/financebench",
    output_dir: str = "results/financebench/rule_gen",
    logs_dir: str = "logs/financebench/agent",
    max_iterations: int = 12,
) -> dict:
    """
    Use a LangChain agent with a structured tool set to iteratively generate,
    validate, and refine span-retrieval rules. Returns a result dict matching
    the rule_gen_llm_coarse output schema, with additional agent metadata.
    """
```

Output schema is identical to `rule_gen_llm_coarse` with extra fields:
`agent_turns`, `merge_accuracy`, `merge_num_correct`, `total_llm_calls`.

---

## Agent Setup

Reuse `src/agent.py` infrastructure:
- `_azure_llm()` for the LLM
- `create_tool_calling_agent` + `AgentExecutor`
- `run_agent` / `astream_events` for token tracking and trace logging
- `max_iterations=max_iterations`, `verbose=True`

---

## Tool Set

### Tool 1 — `summarize_answer_locations()`

**Mandatory first call.** Computes a cross-document pattern table: for each document, applies a fast substring scan to locate where the ground truth answer appears in `doc["texts"]`, then reports structural fields of the matching span.

```
doc_name             page  label           level  bold  all_cap  path_text              answer_snippet
AMCOR_2019_10K       1     section_header  H1     1     0        ""                     Amcor plc
ADOBE_2022Q2_10Q     1     section_header  H2     1     0        ""                     Adobe Inc.
COSTCO_2017_10K      1     text            Body   1     0        ""                     Costco Wholesale
INTEL_2016_10K       1     section_header  H1     1     1        ""                     INTEL CORPORATION
```

If the answer is not found by substring match, the row is marked `NOT_LOCATED` — the agent knows it needs a different approach for that document.

Implementation: no LLM call. Substring match (case-insensitive, stripped) over all span texts.

### Tool 2 — `inspect_answer_context(doc_name)`

Shows the 5 spans before and after the answer span in `doc_name`, with full field annotations. Gives the agent the local document context around the answer without dumping the whole document.

```
[idx=3] text="UNITED STATES"  label=section_header  page=1  bold=1  level=H1  path_text=""
[idx=4] text="SECURITIES AND EXCHANGE COMMISSION"  label=section_header  page=1  bold=1  ...
[idx=5] text="Amcor plc"  label=section_header  page=1  bold=1  level=H1  ← ANSWER HERE
[idx=6] text="(Exact name of registrant as specified in its charter)"  label=text  page=1  ...
[idx=7] text="Ireland / 98-1455367"  label=text  page=1  ...
```

No LLM call.

### Tool 3 — `write_rule(rule_name, code)`

Validates and registers a candidate rule at the tool level before any document testing.

Checks:
1. Function name starts with `rule_`
2. Executes in an empty namespace — rejects if any `NameError` (catches hallucinated helpers)
3. Returns `list` when called on a minimal stub document `{"texts": []}`
4. No top-level imports outside the function body

Returns `"OK: rule registered"` or a specific rejection reason. Registered rules are held in memory for subsequent test calls.

### Tool 4 — `test_rule(rule_name)`

Tests a single registered rule against all documents using **substring match only — no LLM call**.

For each document:
- Apply `rule_name(doc)` → get spans
- Check if ground truth answer appears (case-insensitive substring) in the concatenated retrieved text
- Record: retrieved token count, cost ratio (`retrieved_tokens / total_doc_tokens`), hit (True/False), retrieved text snippet (first 100 chars)

Returns per-doc hit table, aggregate hit rate, and **average cost ratio**. Fast — used freely during iteration.

```
doc_name             hit    retrieved_tokens  cost_ratio  retrieved_snippet
AMCOR_2019_10K       ✓      61                0.0007      "Amcor plc\n\nAmcor plc (Exact name..."
ADOBE_2022Q2_10Q     ✓      34                0.0004      "Adobe Inc."
COSTCO_2017_10K      ✗      0                 0.0000      "(empty)"
INTEL_2016_10K       ✓      23                0.0003      "INTEL CORPORATION"

Hit rate: 3/4 (0.75)   Avg cost ratio: 0.0004  ← lower is better
```

If cost ratio for a rule is high (> 0.05), the agent is prompted to tighten the filter conditions to retrieve fewer but more precise spans.

### Tool 5 — `show_uncovered_docs()`

Returns which documents are NOT covered by any currently registered rule (substring match, no LLM). Gives the agent a cheap "what still needs a rule" signal.

```
Uncovered (no registered rule hits):
  - COSTCO_2017_10K
  - BOEING_2018_10K

Covered: 8/10
```

### Tool 6 — `diagnose_failing_doc(doc_name)`

For a document that is uncovered, shows:
1. What each registered rule actually retrieved from this document (span texts, or "empty")
2. Answer context from `inspect_answer_context(doc_name)` (where the answer is)
3. A structural diff: how this document's answer span differs from documents where rules succeed

```
=== Diagnosis: COSTCO_2017_10K ===

Ground truth: "Costco Wholesale Corporation"

Rule results:
  rule_h1_page1:              retrieved=[]  (no H1 on page 1 matching bold+allcap)
  rule_exact_name_keyword:    retrieved=[]  (no span with 'exact name' on page 1)

Answer location (inspect_answer_context):
  [idx=8] text="Costco Wholesale Corporation"  label=text  page=1  bold=1  level=Body

Structural diff vs. passing docs (AMCOR, ADOBE, INTEL):
  - Those docs: label=section_header, level=H1 or H2
  - This doc:   label=text, level=Body  ← rule_h1_page1 misses Body-level spans
  - Suggested fix: extend rule_h1_page1 to also match bold Body spans on page 1
```

No LLM call.

### Tool 7 — `test_union()`

Tests the union of ALL currently registered rules using **LLM QA + LLM judge** — the true merge accuracy metric. Called sparingly (only when the agent believes coverage is complete or wants a checkpoint).

For each document:
1. Union spans from all registered rules (deduplicated by index in `doc["texts"]`)
2. Sort by `(page_no, structure.level_index)`
3. Compute cost ratio: `len(union_tokens) / total_doc_tokens`
4. Call LLM QA: `retrieved_text → predicted_answer`
5. Call LLM judge: `predicted vs ground_truth → CORRECT/INCORRECT`

Returns **merge accuracy and avg cost ratio**, per-doc results, and for failing docs: retrieved text + predicted answer.

```
Merge accuracy:   9/10 (0.90)
Avg cost ratio:   0.031   ← union tokens / doc tokens, averaged over all docs

Per-doc costs:
  AMCOR_2019_10K:   correct=true   cost_ratio=0.0007
  EBAY_2021_10K:    correct=false  cost_ratio=0.142   ← high cost, wrong answer
  ...

Failing docs:
  EBAY_2021_10K:
    retrieved: "eBay Korea, LLC\n\neBay Inc."
    predicted: "eBay Korea, LLC"
    ground_truth: "eBay Inc."
    → Rule retrieving subsidiary before parent. Need path_text filter.

High-cost rules (avg cost_ratio > 0.05):
  rule_broad_page1: avg_cost=0.18 — consider tightening filter conditions
```

---

## User Prompt

Built per-run. Contains:

1. Question and ground truth table (`doc_name → answer`) for all N documents
2. First 80 spans of each document's `texts` array as JSON
3. The 7 hint types (same wording as `rule_gen_llm_coarse`)
4. Explicit target statement prepended before document content:

```
TARGET:
  1. merge_accuracy >= 0.90  (primary)
  2. avg_cost_ratio as small as possible  (secondary)

merge_accuracy = fraction of documents where the LLM, given the UNION of text
retrieved by ALL your rules, produces the correct answer.

avg_cost_ratio = average over all docs of:
  (tokens in union of retrieved spans) / (total tokens in document)

Each rule should cover as many documents as possible AND retrieve only the
minimal spans needed to contain the answer. A rule that retrieves a whole page
when only one span is needed has unnecessarily high cost.
Start with the most precise rule you can write, verify hit rate and cost with
test_rule(), then add targeted rules only for remaining uncovered documents.
```

---

## Agent Workflow

### Mandatory Turn 1 — Pattern Analysis

The system prompt requires the agent to call `summarize_answer_locations()` first. No rule writing before this call.

### Turn 2 — Plan

Agent identifies the dominant structural pattern (covering the most docs) and plans one broad rule for it. States plan in text before calling `write_rule`.

### Turns 3–N — Generate, Test, Diagnose, Refine

```
write_rule(rule_1_for_pattern_A)
write_rule(rule_2_for_pattern_B)
...
test_rule(rule_1)     ← fast, no LLM
test_rule(rule_2)     ← fast, no LLM
show_uncovered_docs() ← fast, no LLM
diagnose_failing_doc(failing_doc) ← fast, no LLM
write_rule(rule_3_fix_for_failing_doc)
test_rule(rule_3)
show_uncovered_docs()  ← if 0 uncovered →
test_union()           ← first LLM call
```

After `test_union` returns merge accuracy ≥ 0.90, the agent enters a **cost reduction phase**:
- For any rule whose avg `cost_ratio > 0.05` (from `test_rule`), attempt to tighten its filter conditions
- After each tightening: call `test_rule` to verify hit rate is maintained, then `test_union` to confirm accuracy is not degraded
- Stop when no further cost reduction is possible without losing accuracy, or `max_iterations` is reached

---

## System Prompt

```
You are a document rule engineer. Generate Python span-retrieval rules for financial documents.

DOCUMENT STRUCTURE
[same as rule_gen_agent_coarse_claude.py — text, label, page_no, bold, size,
structure.level, structure.path_text, table_data.cells]

RULE INTERFACE
  def rule_<name>(doc: dict) -> list[dict]:
      """One-line description."""
      return [span for span in doc["texts"] if ...]
  Rules must be self-contained (import inside function). Never raise. Return [] on failure.

PRIMARY OBJECTIVES (in order of priority)
  1. merge_accuracy >= 0.90
     Fraction of docs where the LLM, given the UNION of text retrieved by ALL
     your rules, produces the correct answer.
  2. avg_cost_ratio as small as possible
     avg_cost_ratio = mean over all docs of:
       (tokens in union of retrieved spans) / (total tokens in document)
     Lower is better — rules should retrieve the minimal spans needed.
  3. Small number of rules (prefer 2–4 over 10+)

  A rule returning [] for a doc contributes nothing. Only rules that retrieve
  the answer-containing span drive accuracy up. But retrieving entire sections
  when only one span is needed wastes cost — aim for the tightest filter that
  still captures the answer.

RULE WRITING STRATEGY
  - Each rule should cover as many sampled documents as possible.
  - Write rules as precisely as possible: filter by page_no, label, level,
    bold, path_text. Fewer spans returned = lower cost.
  - After test_rule(), check BOTH hit rate AND avg cost_ratio.
    If cost_ratio > 0.05, tighten filter conditions to retrieve fewer spans.
  - Never modify a rule already hitting its documents. Add new rules for gaps.
  - After achieving merge_accuracy >= 0.90, enter cost reduction phase:
    for each rule with cost_ratio > 0.05, tighten conditions and confirm
    accuracy is maintained via test_union() before keeping the change.
  - Do not define helper functions outside the rule function body.
  - Prefer path_text anchoring over keyword-only matching.

MANDATORY WORKFLOW — follow this order exactly:
1. Call summarize_answer_locations() FIRST. Do not write any rule before this.
2. Identify the dominant structural pattern. Write ONE precise rule for it.
3. write_rule() → test_rule() — check BOTH hit rate and cost_ratio.
4. show_uncovered_docs() → diagnose_failing_doc() for each uncovered doc.
5. Write targeted rule for each uncovered structural variant.
6. Repeat 3–5 until show_uncovered_docs() = 0.
7. Call test_union() — check merge_accuracy AND avg_cost_ratio.
8. If merge_accuracy < 0.90: fix failing docs, repeat from 4.
9. If merge_accuracy >= 0.90 but cost high: tighten high-cost rules,
   call test_union() to verify accuracy is preserved, repeat until cost stable.

ADDITIONAL RULES:
- Each rule must have a unique name. Call write_rule() for all new/updated rules.
- Do not define helper functions outside the rule function body.
- Prefer path_text anchoring over keyword-only matching for structural questions.
```

---

## Output Files

Same as `rule_gen_llm_coarse` and `rule_gen_agent_coarse_claude`:

| Artifact | Path |
|---|---|
| Result summary | `results/financebench/rule_gen/{question_slug}_{n}_{timestamp}.json` |
| Rule folder | `rules/llm/financebench/{question_slug}_{n}_llm/` |
| Rule files | `rules/llm/financebench/{question_slug}_{n}_llm/rule_*.py` |
| Trace log | `logs/financebench/agent/{question_slug}_{n}docs_{timestamp}.txt` |

Extra fields in result JSON (vs. `rule_gen_llm_coarse`):

| Field | Description |
|---|---|
| `agent_turns` | Number of agent reasoning steps |
| `merge_accuracy` | Final merge accuracy from last `test_union` call |
| `merge_num_correct` | Number of docs correct under union |
| `avg_cost_ratio` | Final avg cost ratio — mean of (union retrieved tokens / total doc tokens) across all docs |
| `total_llm_calls` | Total LLM QA+judge calls inside `test_union` across all invocations |
| `agent_input_tokens` | Input tokens consumed by agent reasoning turns (via `astream_events`) |
| `agent_output_tokens` | Output tokens generated by agent reasoning turns |
| `judge_input_tokens` | Input tokens consumed by QA + judge calls inside `test_union` |
| `judge_output_tokens` | Output tokens generated by QA + judge calls inside `test_union` |
| `total_input_tokens` | `agent_input_tokens + judge_input_tokens` |
| `total_output_tokens` | `agent_output_tokens + judge_output_tokens` |
| `rules[].hit_rate` | Substring hit rate — fraction of docs where ground truth appears in retrieved text (no LLM) |
| `rules[].avg_cost_ratio` | Average cost ratio for this rule alone across all docs |

---

## Trace Log Format

One file per agent run. Written to:
```
logs/financebench/agent/{question_slug}_{n}docs_{timestamp}.txt
```

Format matches the existing trace style in `logs/` (e.g. `logs/3M_2023Q2_10Q_1_trace.txt`):

```
Question: {question}
Num docs: {n}
Timestamp: {YYYY-MM-DDTHH:MM:SSZ}
Latency: {X.XXX}s
Agent turns: {K}
Input tokens: {N}
Output tokens: {N}
Total LLM calls (union eval): {N}

=== Agent Turn 1: summarize_answer_locations ===
Input: {}
Result:
doc_name             page  label           level  bold  path_text    answer_snippet
AMCOR_2019_10K       1     section_header  H1     1     ""           Amcor plc
ADOBE_2022Q2_10Q     1     section_header  H2     1     ""           Adobe Inc.
...

=== Agent Turn 2: write_rule ===
Input: {"rule_name": "rule_h1_page1", "code": "def rule_h1_page1(doc): ..."}
Result: OK: rule registered

=== Agent Turn 3: test_rule ===
Input: {"rule_name": "rule_h1_page1"}
Result:
doc_name             hit  retrieved_tokens  snippet
AMCOR_2019_10K       ✓    61                Amcor plc...
ADOBE_2022Q2_10Q     ✓    34                Adobe Inc....
COSTCO_2017_10K      ✗    0                 (empty)
...
Hit rate: 8/10 (0.80)

=== Agent Turn 4: show_uncovered_docs ===
Input: {}
Result:
Uncovered: COSTCO_2017_10K, BOEING_2018_10K
Covered: 8/10

=== Agent Turn 5: diagnose_failing_doc ===
Input: {"doc_name": "COSTCO_2017_10K"}
Result:
Rule results:
  rule_h1_page1: (empty) — no H1 bold span on page 1
Answer location:
  [idx=8] text="Costco Wholesale Corporation"  label=text  page=1  bold=1  level=Body
Structural diff vs. passing docs:
  This doc: label=text, level=Body | Passing docs: label=section_header, level=H1/H2

=== Agent Turn 6: write_rule ===
Input: {"rule_name": "rule_bold_body_page1", "code": "..."}
Result: OK: rule registered

=== Agent Turn 7: show_uncovered_docs ===
Result: Covered: 10/10 — call test_union() now

=== Agent Turn 8: test_union ===
Input: {}
Result:
Merge accuracy: 10/10 (1.00)
All documents answered correctly.

=== Final Rules ===
rule_h1_page1:       hit_rate=8/10  file=rules/llm/financebench/.../rule_h1_page1.py
rule_bold_body_page1: hit_rate=2/10  file=rules/llm/financebench/.../rule_bold_body_page1.py

=== Summary ===
Merge accuracy:        1.00  (10/10)
Avg cost ratio:        0.0006
Total LLM calls:       20  (1 test_union × 10 docs × 2 calls each)
Agent input tokens:    3710   ← agent reasoning turns only
Agent output tokens:   890
Judge input tokens:    1110   ← QA + judge calls inside test_union
Judge output tokens:   215
Total input tokens:    4820
Total output tokens:   1105
Latency:               38.4s
```

---

## Edge Cases

| Situation | Behavior |
|---|---|
| `write_rule` rejects code (NameError, no list return) | Tool returns rejection reason; agent must fix before testing |
| Answer not locatable in any doc by substring | `summarize_answer_locations` marks those docs `NOT_LOCATED`; agent told to use `inspect_answer_context` |
| `test_union` called with no registered rules | Returns `merge_accuracy=0.0` immediately |
| Agent reaches `max_iterations` | Save all registered rules; record final `test_union` result |
| Rule overwrites existing file in rules folder | Overwrite — latest version wins |

---

## Relation to Other Modules

| Module | Role |
|---|---|
| `src/rule_gen_llm_coarse.py` | Single-pass; no verification |
| `src/rule_gen_agent_coarse_claude.py` | Agent loop; `inspect_doc` + `test_rules` (LLM per rule); predecessor |
| `src/rule_gen_agent_langchain.py` | This module — pattern-first, fast feedback, diagnosis-driven |
| `src/rule_apply_merge.py` | Consumes rules unchanged |
| `src/agent.py` | LangChain agent infrastructure reused here |
