# Rule Refinement — `src/rule_refine.py`

---

## Overview

Given a full set of rules for a question, this module selects a **minimal-cost subset** whose merge accuracy matches the target (the merge accuracy of all rules). It requires no additional LLM calls beyond what is already pre-computed — cost estimation is free (token counting only), and merge accuracy is evaluated by calling the LLM on the union of retrieved spans.

The algorithm has three phases: cost computation, exponential search to find a sufficient prefix, and backward linear scan to prune redundant rules.

---

## Relation to Optimization Problem

This is a greedy approximate solution to Problem 1 (minimize cost subject to maintaining merge accuracy). It exploits two properties:

- **Cost is free to compute** — no LLM needed, just token counting
- **acc(S) is monotone** — if S achieves target accuracy, any superset also achieves it

The algorithm trades optimality for simplicity: it searches within cost-sorted rules (greedy ordering) and prunes via a single linear pass.

---

## Function Interface

```python
def rule_refine(
    rule_names: list[str],        # full set of candidate rule names
    target_accuracy: float,       # merge accuracy to match (from pre-computed eval)
    question: str,                # full question text
    question_slug: str,           # e.g. "what_is_the_registrants_exact_name_10"
    documents: list[dict],        # loaded document JSONs
    ground_truth: dict,           # { "DOCNAME.pdf": "answer string" }
    rules_dir: str = "rules/llm/financebench",
    output_dir: str = "results/llm_rule_refine",
    processing_dir: str = "data/financebench/processing",
    model_name: str = "gpt54",
) -> dict:
    """
    Select a minimal-cost subset of rules whose merge accuracy matches target_accuracy.

    Returns a result dict with selected rule names, merge accuracy, cost ratio,
    latency, and per-document results.
    """
```

---

## Algorithm

### Step 0 — Compute cost of every rule

For each rule $r_i$ and each document $d_j$:

```python
cost_ij = retrieved_token_count(r_i, d_j) / total_token_count(d_j)
```

Apply each rule to every document (no LLM call). Count tokens with tiktoken (`cl100k_base`), falling back to `len(text.split()) * 1.3`.

Average cost per rule:

```python
avg_cost_i = mean(cost_ij for all j)
```

Retrieve target merge accuracy from the pre-computed eval result (passed as `target_accuracy`).

### Step 1 — Sort rules by cost (ascending)

```python
sorted_rules = sorted(rule_names, key=lambda r: avg_cost_r)
# sorted_rules[0] is cheapest, sorted_rules[-1] is most expensive
```

### Step 2 — Exponential search for minimal sufficient prefix

Check prefixes of increasing size: 1, 2, 4, 8, ... until merge accuracy ≥ target.

```python
k = 1
while k <= len(sorted_rules):
    candidate = sorted_rules[:k]
    acc = evaluate_merge_accuracy(candidate, documents, ground_truth, question)
    if acc >= target_accuracy:
        break
    k = min(k * 2, len(sorted_rules))
    if k == len(sorted_rules) and acc < target_accuracy:
        # use all rules — target not achievable with fewer
        candidate = sorted_rules
        break
```

After exponential search, `candidate = sorted_rules[:k]` is the first prefix that meets the target.

**LLM calls in this step:** $O(\log m)$ merge evaluations, each calling LLM once per document per candidate set.

### Step 3 — Backward linear scan (pruning)

Scan from the most expensive rule in `candidate` (index $k-1$) down to index 1. At each step, try removing the current rule and check if accuracy is maintained:

```python
refined = list(candidate)  # copy of top-k
for i in range(len(refined) - 1, 0, -1):   # from k-1 down to 1 (never remove index 0)
    rule_to_test = refined[i]
    subset = refined[:i] + refined[i+1:]    # remove rule at position i
    acc = evaluate_merge_accuracy(subset, documents, ground_truth, question)
    if acc >= target_accuracy:
        refined.remove(rule_to_test)        # safe to drop
    # else: keep it
```

Return `refined` as the final selected rule set.

**LLM calls in this step:** at most $k - 1$ merge evaluations.

**Total LLM calls:** $O(\log m + k)$ where $k \leq m$.

---

## `evaluate_merge_accuracy` Helper

```python
def evaluate_merge_accuracy(rule_names, documents, ground_truth, question) -> float:
    """
    For each document:
      1. Apply each rule → get spans
      2. Union spans, deduplicate by index, sort by (page_no, level_index)
      3. Concatenate span texts → retrieved_text
      4. Call LLM with same prompt as rule_apply_merge
      5. Call LLM judge to compare predicted vs ground_truth
    Return fraction correct across all documents.
    """
```

Uses same system/user prompts as `rule_apply_merge` and `eval_rule`:
- **QA prompt:** "You are a financial document QA assistant. Answer using only the passage. Reply NOT FOUND if insufficient. Return only the answer."
- **Judge prompt:** "Judge whether predicted answer is semantically equivalent to ground truth. Reply CORRECT or INCORRECT."

---

## Output Format

### Result file

Written to: `{output_dir}/{question_slug}_refine.json`

```json
{
  "question": "What is the registrant's exact name?",
  "question_slug": "what_is_the_registrants_exact_name_10",
  "timestamp": "2026-04-29T10:00:00Z",
  "model": "gpt54",
  "num_documents": 10,
  "target_accuracy": 0.9,
  "all_rules_count": 25,
  "selected_rules_count": 4,
  "selected_rules": [
    "rule_cover_page_h1_before_exact_name_text",
    "rule_exact_name_caption_parent",
    "rule_cover_page_name_all_caps_or_title_case",
    "rule_item1_first_company_sentence"
  ],
  "merge_accuracy": 0.9,
  "avg_cost_ratio": 0.0021,
  "avg_cost_ratio_all_rules": 0.0187,
  "cost_reduction_ratio": 0.888,
  "avg_latency_seconds": 1.34,
  "total_llm_calls": 11,
  "exponential_search_steps": 4,
  "pruning_steps": 7,
  "per_doc": [
    {
      "doc_name": "AMCOR_2019_10K",
      "predicted": "Amcor plc",
      "ground_truth": "Amcor plc",
      "correct": true,
      "retrieved_tokens": 42,
      "total_doc_tokens": 84134,
      "cost_ratio": 0.0005,
      "latency_seconds": 1.2
    }
  ]
}
```

### Selected rules folder

Copy selected rule `.py` files to:

```
results/llm_rule_refine/
└── {question_slug}/
    ├── rule_cover_page_h1_before_exact_name_text.py
    ├── rule_exact_name_caption_parent.py
    └── ...
```

Format is identical to `rules/llm/financebench/{question_slug}_llm/`.

---

## Full Directory Structure

```
results/
└── llm_rule_refine/
    ├── what_is_the_registrants_exact_name_10/
    │   ├── rule_cover_page_h1_before_exact_name_text.py
    │   └── rule_exact_name_caption_parent.py
    ├── what_is_total_revenue_10/
    │   └── rule_income_statement_revenue_row.py
    ├── what_is_the_registrants_exact_name_10_refine.json
    ├── what_is_total_revenue_10_refine.json
    └── summary.json
```

### Summary file

`results/llm_rule_refine/summary.json` — one entry per question:

```json
[
  {
    "question": "What is the registrant's exact name?",
    "question_slug": "what_is_the_registrants_exact_name_10",
    "target_accuracy": 0.9,
    "merge_accuracy": 0.9,
    "all_rules_count": 25,
    "selected_rules_count": 4,
    "avg_cost_ratio_selected": 0.0021,
    "avg_cost_ratio_all": 0.0187,
    "cost_reduction_ratio": 0.888,
    "total_llm_calls": 11
  }
]
```

---

## Key Metrics

| Metric | Description |
|---|---|
| `merge_accuracy` | Accuracy of selected rules on sampled docs — should equal `target_accuracy` |
| `avg_cost_ratio` | Average fraction of doc tokens retrieved by selected rules (union) |
| `avg_cost_ratio_all_rules` | Same metric for the full rule set (baseline) |
| `cost_reduction_ratio` | `1 - avg_cost_ratio / avg_cost_ratio_all_rules` — fraction of cost saved |
| `total_llm_calls` | Total LLM evaluations during search + pruning |

---

## Edge Cases

| Situation | Behavior |
|---|---|
| Single rule achieves target accuracy | Return that rule; skip pruning |
| No subset achieves target accuracy | Return all rules; log warning |
| All rules needed (pruning removes nothing) | Return full sorted list; `cost_reduction_ratio = 0` |
| Rule file missing | Skip that rule; log warning |
| Output file already exists | Overwrite |

---

## Relation to Other Modules

| Module | Role |
|---|---|
| `src/rule_gen_llm_coarse.py` | Generates the candidate rules consumed here |
| `src/rule_apply_merge.py` | Merge evaluation logic reused in `evaluate_merge_accuracy` |
| `src/eval_rule.py` | Judge logic reused for correctness checking |
| `results/eval_merge_all/` | Source of `target_accuracy` per question |
| `rules/llm/financebench/` | Source of candidate rule `.py` files |
