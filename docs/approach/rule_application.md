# Rule Application

This document describes the rule application strategies in the LSF codebase. Each takes a loaded document and one or more span-retrieval rules, retrieves matching spans, and calls an LLM to answer a question from the retrieved text.

Three production strategies are documented:

1. **Merge** — apply a fixed rule set, take the union, ask the LLM once.
2. **Merge + Default (fallback)** — same as Merge, but if a cheap gate model judges the retrieved text insufficient, retry with the full rule pool.
3. **Cost-Descent** — like Default, but on the hit-path it keeps shrinking the context toward the *cheapest* rules (cheap-gated halving) so the expensive model reads the smallest still-sufficient subset. See [Strategy 3](#strategy-3--cost-descent) and the full design/analysis in [`rule_apply_descent.md`](rule_apply_descent.md).

A fourth utility, **Individual**, is kept in the codebase for debugging and per-rule diagnostics; it is not a deployment strategy. See the [Debugging utility](#debugging-utility--individual) section at the end.

---

## Strategy comparison

| Aspect | Merge | Merge + Default (fallback) | Cost-Descent |
|--------|-------|---------------------------|--------------|
| Code | `src/rule_apply/merge.py` | `src/rule_apply/default.py` | `src/rule_apply/descent.py` |
| Rules input | `rule_names: list[str]` (the refined subset) | refined subset + full pool | refined subset + full pool |
| Retrieval | Apply all rules, union + dedupe spans, LLM answers once | Merge over refined subset first; on gate "NO" verdict, re-merge over full pool and answer from that | Cost-sort the refined rules; halve toward the cheapest while the gate says YES; answer from the smallest passing subset (full-pool fallback if even the full refined set fails) |
| gpt54 calls per doc | 1 | 1 (hit) / 1 (miss) | **1** (always) |
| gpt54mini gate calls per doc | 0 | 1 | `1 + ⌊log₂ n⌋` (hit) / 1 (miss) |
| gpt54 context (cost) | full refined retrieval | refined (hit) / full pool (miss) | **smallest passing subset ≤ refined** (hit) / full pool (miss) |
| Best when | You trust the refined subset to cover unseen docs | You want refined-set cost on most docs and full-pool safety net on the rest | You want to push cost below the refined-set retrieval when the answer concentrates in a few cheap rules |
| Reference results | sAcc/uAcc as reported in `rule_refinement.md` per selector | uAcc 0.892 at cost_u 0.030 (single cluster); uAcc 0.940 at cost_u 0.009 (multi cluster) | *not yet benchmarked* |

> **Cost accounting (Cost-Descent):** both gpt54mini and gpt54 tokens are logged, but the reported **average cost ratio uses gpt54 tokens only** — the single gpt54 context size divided by doc tokens. gpt54mini gate tokens are excluded from the cost metric.

---

## Strategy 1 — Merge

**Code:** `src/rule_apply/merge.py`

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

```json
{
  "rule_names": ["rule_exact_name_parent_h1", "rule_page1_first_h1"],
  "rule_set_slug": "rule_exact_name_parent_h1__rule_page1_first_h1",
  "strategy": "merge",
  "question": "...",
  "question_slug": "...",
  "doc_name": "...",
  "predicted_answer": "...",
  "rules_with_hits": ["rule_exact_name_parent_h1"],
  "rules_with_no_hits": ["rule_page1_first_h1"],
  "num_spans_before_dedup": 5,
  "num_spans_after_dedup": 3,
  "retrieved_token_count": 245,
  "retrieved_spans": [...],
  "retrieved_text": "3M COMPANY",
  "input_tokens": 312,
  "output_tokens": 8,
  "latency_seconds": 2.1
}
```

---

## Strategy 2 — Merge + Default (refined-with-fallback)

**Code:** `src/rule_apply/default.py`

### Description

A deployment-time strategy that uses **Merge** as its retrieval primitive but adds a safety net: apply the refined rule subset first; if a cheap gate model judges the retrieved text insufficient to answer the question, re-merge over the full LLM-coarse pool and answer from that instead. Combines the per-doc cost of a refined set with the recall of the full pool.

Rule source: any refined subset (typically from `select_rules_pareto_v2` or the agentic selector) + the corresponding full LLM-coarse pool.

### Algorithm (per doc at inference)

1. Merge over the **refined rule subset** → `retrieved_text_refined`.
2. Ask **gpt54mini**: "Does this passage contain enough information to answer the question?"
3. If YES → answer with **gpt54** on `retrieved_text_refined`.
4. If NO → re-merge over the **full pool** → `retrieved_text_full`; answer with gpt54 on that.

### Interface

```python
def apply_with_fallback(
    document: dict,
    refined_rule_names: list[str],
    full_pool_rule_names: list[str],
    question_slug: str,
    question: str,
    rules_dir: str = "rules/financebench",
    relevance_model: str = "gpt54mini",
    answer_model: str = "gpt54",
) -> dict
```

### Per-doc output schema (additions over Merge)

```json
{
  "strategy": "default_with_fallback",
  "used_fallback": false,
  "relevance_verdict": "YES",
  "relevance_input_tokens": 312,
  "relevance_output_tokens": 1,
  "answer_input_tokens": 312,
  "answer_output_tokens": 14
}
```

### Results

**FinanceBench, single cluster (refined subset = p_v2 selection):**

| uAcc | cost_u | Mean fallback rate |
|-----:|--------:|-------------------:|
| **0.892** | **0.030** | 11.8% |

Matches the full-pool uAcc (0.892) at 18% of base retrieval cost (0.030 vs 0.169). The gate fires on ~6 of 50 unsampled docs per question. Recovers the entire refined→base generalization gap at ~3× the refined-only retrieval cost (still 5.6× cheaper than always applying the full pool).

**FinanceBench, multi cluster (refined subset = agentic selection, 12 questions, 68 unsampled docs):**

| uAcc | cost_u | Mean fallback rate |
|-----:|--------:|-------------------:|
| **0.940** | **0.009** | 11.2% |

Beats the full-pool baseline (0.935) at 6.5× lower cost.

---

## Strategy 3 — Cost-Descent

**Code:** `src/rule_apply/descent.py` · **Design + analysis:** [`rule_apply_descent.md`](rule_apply_descent.md)

### Description

A deployment-time strategy that, like Default, retrieves over a refined subset and gates with gpt54mini — but instead of feeding the *whole* refined retrieval to gpt54, it shrinks the context toward the **cheapest** rules while the gate still finds the answer, then answers from the smallest passing subset. The expensive model is called **exactly once**; only the size of its context changes. Reduces to Default at both ends (single-rule sets, immediate gate failure, or refined-set failure → full-pool fallback).

Rule source: any refined subset + the corresponding full pool — **or** the rule set an agent returns directly in the rule-end-to-end strategy (`agentic_rule_full_data`), in which case the applied set and fallback pool can be the same folder.

### Algorithm (per doc at inference)

1. Sort refined rules by **per-doc cost** ascending — a rule's cost = the tokens it retrieves on *this* document. `top-k` = the `k` cheapest rules (nested: `top-1 ⊂ … ⊂ top-n`).
2. Retrieve `top-n` (all refined). Ask **gpt54mini** if it contains the answer.
   - **NO** → re-merge over the **full pool**, answer with **gpt54**. (identical to Default's miss-path)
   - **YES** → descend.
3. Test `top-n/2`. If gpt54mini says YES, accept it and recurse to `top-n/4`, … down to `top-1`. The first **NO** stops the descent.
4. Answer once with **gpt54** on the **last YES level** (the smallest passing subset).

### Interface

```python
def apply_with_descent(
    document, question,
    refined_rules: list[str], all_rules: list[str],
    rule_folder: Path, fallback_folder: Path | None = None,   # refined / full-pool dirs
    relevance_model="gpt54mini", qa_model="gpt54",
) -> dict
```

Wired into `src/pipeline.py` as `apply_strategy="descent"` (requires `refine_strategy != "none"`, like `default`).

### Per-doc output (additions over Default)

```json
{
  "strategy": "descent",
  "used_fallback": false,
  "final_k": 2,
  "final_n": 8,
  "descent_trace": [
    {"k": 8, "tokens": 412, "verdict": "yes"},
    {"k": 4, "tokens": 210, "verdict": "yes"},
    {"k": 2, "tokens": 96,  "verdict": "yes"},
    {"k": 1, "tokens": 38,  "verdict": "no"}
  ],
  "retrieved_token_count": 96,
  "relevance_input_tokens": 740, "relevance_output_tokens": 4
}
```

`retrieved_token_count` is the gpt54 context size (here `top-2` = 96 tokens) — the only quantity that enters the cost ratio. `final_k/final_n` records how far it shrank.

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
| Output file exists | Read, append, write back |

---

## Output directory layout

```
results/financebench/rule_run/
└── merge/
    └── {question_slug}/
        └── {rule_set_slug}_merge.json
```

(`default.py` writes per-doc JSON into a caller-owned directory; the default driver is the eval script that invokes it.)

---

## Debugging utility — Individual

**Code:** `src/rule_apply/individual.py`

Not a production deployment strategy. Kept in the codebase for debugging and per-rule diagnostics: applies **one** named rule to **one** document and runs the LLM on whatever that single rule retrieves. Useful when you want to know what a single rule contributes in isolation — e.g. while drafting a new rule, measuring per-rule coverage, or investigating why Merge picks up unexpected spans.

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
4. Call LLM (same prompt as Merge).
5. Append result record to `{output_dir}/{question_slug}/{rule_name}_individual.json`.

Output schema is the same as Merge's, except `strategy = "individual"` and `rule_name` replaces `rule_names`/`rule_set_slug`. Missing rule file raises `FileNotFoundError` (Merge skips with a warning — Individual is meant to fail loud when you ask for a specific rule that isn't there).
