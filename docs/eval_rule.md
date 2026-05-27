# Rule Evaluation — `src/eval_rule.py`

---

## Overview

This module evaluates a single rule by:
1. Loading its predictions from the `rule_run` results folder
2. Comparing each prediction against ground truth using **LLM-as-a-judge**
3. Computing accuracy and cost ratio across all documents
4. Storing per-(rule, document) results and aggregate metrics

---

## Inputs

| Argument | Type | Description |
|---|---|---|
| `rule_name` | string | Name of the rule, e.g. `rule_exact_name_parent_h1` |
| `doc_names` | list[string] | Document names to evaluate, e.g. `["AMCOR_2019_10K", "ADOBE_2022Q2_10Q"]` — no `.pdf` suffix |
| `question` | string | Full question text |
| `question_slug` | string | Slug used in folder names, e.g. `what_is_the_registrants_exact_name_10` |
| `model_name` | string | LLM for judging, default `"gpt54"` |
| `rule_run_dir` | string | Base dir for rule run results, default `"results/financebench/rule_run/individual"` |
| `processing_dir` | string | Original document JSONs, default `"data/financebench/processing"` |
| `labels_file` | string | Ground truth file, default `"data/financebench/sample_labels.json"` |
| `output_dir` | string | Where to write eval results, default `"results/financebench/eval"` |

---

## Function Interface

```python
def eval_rule(
    rule_name: str,
    doc_names: list[str],
    question: str,
    question_slug: str,
    model_name: str = "gpt54",
    rule_run_dir: str = "results/financebench/rule_run/individual",
    processing_dir: str = "data/financebench/processing",
    labels_file: str = "data/financebench/sample_labels.json",
    output_dir: str = "results/financebench/eval",
) -> dict:
    """
    Evaluate a rule against ground truth using LLM-as-a-judge.
    Returns aggregate metrics and per-document results.
    """
```

---

## Step-by-Step Logic

### Step 1 — Load rule run predictions

Read:
```
{rule_run_dir}/{question_slug}/{rule_name}_individual.json
```

This is a JSON array. Index it by `doc_name` for fast lookup:
```python
predictions = { record["doc_name"]: record for record in records }
```

If a doc from `doc_names` is missing from the predictions file, record it as `{"predicted_answer": null, "retrieved_token_count": 0}`.

### Step 2 — Load ground truth

```python
labels = json.load(open(labels_file))
# labels keyed by "DOCNAME.pdf" → { question: answer }
ground_truth = labels.get(doc_name + ".pdf", {}).get(question, None)
```

### Step 3 — LLM-as-a-judge

For each document, call the LLM to compare `predicted_answer` vs `ground_truth`:

**System prompt:**
```
You are an answer equivalence judge for a financial document QA system.
You will be given a question, a predicted answer, and a ground truth answer.
Judge whether the predicted answer is correct — meaning semantically equivalent
to the ground truth, ignoring minor formatting differences.

Equivalence rules:
- Treat "2017" and "year 2017" as the same
- Treat "$4.5 billion" and "4,500 million" as the same if numerically equal
- Treat "NYSE" and "New York Stock Exchange" as the same
- Ignore leading/trailing whitespace, punctuation, and capitalization differences
- If the predicted answer is "NOT FOUND" or null, always judge as incorrect

Reply with exactly one word: CORRECT or INCORRECT
```

**User prompt:**
```
Question: {question}
Ground Truth: {ground_truth}
Predicted: {predicted_answer}
```

Record `judge_input_tokens`, `judge_output_tokens`, `judge_latency_seconds` per call.

### Step 4 — Compute cost ratio

For each document, compute:
```python
# Total tokens in original document (all span texts concatenated)
total_doc_tokens = count_tokens("\n".join(s["text"] for s in doc["texts"]))

# Tokens retrieved by the rule
retrieved_tokens = prediction["retrieved_token_count"]

cost_ratio = retrieved_tokens / total_doc_tokens if total_doc_tokens > 0 else 0.0
```

Aggregate: `avg_cost_ratio = mean(cost_ratio for all docs)`

### Step 5 — Write results

#### Per-document result record

```json
{
  "rule_name": "rule_exact_name_parent_h1",
  "question_slug": "what_is_the_registrants_exact_name_10",
  "question": "What is the registrant's exact name?",
  "doc_name": "AMCOR_2019_10K",
  "strategy": "individual",
  "predicted_answer": "Amcor plc",
  "ground_truth": "Amcor plc",
  "correct": true,
  "retrieved_token_count": 12,
  "total_doc_tokens": 9840,
  "cost_ratio": 0.00122,
  "judge_input_tokens": 85,
  "judge_output_tokens": 1,
  "judge_latency_seconds": 0.9,
  "rule_apply_latency_seconds": 2.1
}
```

`rule_apply_latency_seconds` is the `latency_seconds` from the original rule_run result record — the time taken by the LLM to generate the predicted answer (not the judge call).

#### Aggregate result file

Written to:
```
{output_dir}/{question_slug}/{rule_name}_eval.json
```

```json
{
  "rule_name": "rule_exact_name_parent_h1",
  "question_slug": "what_is_the_registrants_exact_name_10",
  "question": "What is the registrant's exact name?",
  "strategy": "individual",
  "num_documents": 10,
  "accuracy": 0.8,
  "avg_cost_ratio": 0.00134,
  "avg_retrieved_token_count": 13.2,
  "total_eval_latency_seconds": 11.2,
  "avg_rule_apply_latency_seconds": 2.3,
  "avg_judge_latency_seconds": 0.9,
  "total_judge_input_tokens": 850,
  "total_judge_output_tokens": 10,
  "total_judge_latency_seconds": 9.1,
  "per_document": [
    {
      "doc_name": "AMCOR_2019_10K",
      "predicted_answer": "Amcor plc",
      "ground_truth": "Amcor plc",
      "correct": true,
      "retrieved_token_count": 12,
      "total_doc_tokens": 9840,
      "cost_ratio": 0.00122,
      "judge_input_tokens": 85,
      "judge_output_tokens": 1,
      "judge_latency_seconds": 0.9,
      "rule_apply_latency_seconds": 2.1
    }
  ]
}
```

---

## Output Directory Structure

```
results/financebench/
└── eval/
    └── {question_slug}/
        ├── {rule_name_1}_eval.json
        ├── {rule_name_2}_eval.json
        └── ...
```

Example:
```
results/financebench/eval/
└── what_is_the_registrants_exact_name_10/
    ├── rule_exact_name_parent_h1_eval.json
    ├── rule_cover_page_bold_header_eval.json
    ├── rule_page1_first_h1_not_form_or_commission_eval.json
    └── ...
```

---

## Metrics

| Metric | Formula | Description |
|---|---|---|
| `accuracy` | correct / num_documents | Fraction of documents where predicted answer == ground truth |
| `avg_cost_ratio` | mean(retrieved_tokens / total_doc_tokens) | How much of the document the rule reads on average |
| `avg_retrieved_token_count` | mean(retrieved_token_count) | Average tokens returned by the rule |
| `avg_rule_apply_latency_seconds` | mean(rule_apply_latency_seconds) | Average time the rule's answer-generation LLM call took |
| `avg_judge_latency_seconds` | mean(judge_latency_seconds) | Average time the judge LLM call took per document |
| `total_eval_latency_seconds` | sum of all judge + rule_apply latencies | Total wall-clock time for the full eval run |

**Cost ratio interpretation:** A rule with cost_ratio = 0.001 retrieves 0.1% of the document on average. Lower is better assuming accuracy is maintained. A ratio of 1.0 means the rule returns the entire document (no filtering benefit).

---

## Token Counting

Use `tiktoken` if available (same encoder as the LLM). Fall back to `len(text.split()) * 1.3` if not installed. Be consistent — use the same method for both `retrieved_token_count` and `total_doc_tokens`.

---

## Edge Cases

| Situation | Behavior |
|---|---|
| Prediction file missing for a rule | Raise `FileNotFoundError` with expected path |
| Doc not in prediction file | Record as `predicted_answer=null`, `correct=false`, `retrieved_token_count=0` |
| Doc not in ground truth labels | Record as `ground_truth=null`, `correct=false`, skip judge call |
| Judge returns neither CORRECT nor INCORRECT | Record as `correct=false`, log a warning |
| Original doc JSON not found (for token count) | Use `total_doc_tokens=null`, `cost_ratio=null` |

---

## Test

**File:** `test/test_eval_rule.py`

Tests all rules under `rules/financebench/what_is_the_registrants_exact_name_10/` using:
- Documents: `doc_names` from the most recent rule_gen result file in `results/financebench/rule_gen/` matching the question slug
- Question: `"What is the registrant's exact name?"`
- Question slug: `"what_is_the_registrants_exact_name_10"`

```python
# test/test_eval_rule.py
import os, json, glob
from src.eval_rule import eval_rule

QUESTION = "What is the registrant's exact name?"
QUESTION_SLUG = "what_is_the_registrants_exact_name_10"
RULES_DIR = "rules/financebench/what_is_the_registrants_exact_name_10"

# Load doc_names from most recent rule_gen result
gen_files = sorted(glob.glob(f"results/financebench/rule_gen/*{QUESTION_SLUG}*"))
# fall back to the slug without the _10 suffix
if not gen_files:
    gen_files = sorted(glob.glob("results/financebench/rule_gen/what_is_the_registrants_exact_name_*.json"))
latest = json.load(open(gen_files[-1]))
doc_names = latest["doc_names"]

# Eval all rules
rule_files = sorted(glob.glob(f"{RULES_DIR}/rule_*.py"))
results = []
for rule_file in rule_files:
    rule_name = os.path.splitext(os.path.basename(rule_file))[0]
    print(f"Evaluating {rule_name} ...")
    result = eval_rule(
        rule_name=rule_name,
        doc_names=doc_names,
        question=QUESTION,
        question_slug=QUESTION_SLUG,
    )
    results.append({
        "rule_name": rule_name,
        "accuracy": result["accuracy"],
        "avg_cost_ratio": result["avg_cost_ratio"],
    })
    print(f"  accuracy={result['accuracy']:.2f}  cost_ratio={result['avg_cost_ratio']:.4f}")

# Print summary sorted by accuracy desc
results.sort(key=lambda r: -r["accuracy"])
print("\n=== Summary ===")
for r in results:
    print(f"{r['rule_name']:60s}  acc={r['accuracy']:.2f}  cost={r['avg_cost_ratio']:.4f}")
```

---

## Relation to Other Modules

| Module | Role |
|---|---|
| `src/rule_gen/llm_coarse.py` | Generates rules and writes `doc_names` to rule_gen result |
| `src/rule_apply/individual.py` | Runs rules and writes predictions to rule_run folder |
| `src/eval_rule.py` | This module — reads predictions, judges, writes eval results |
| `test/test_eval_rule.py` | Runs eval on all rules for one question |
