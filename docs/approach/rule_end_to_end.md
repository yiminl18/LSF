# Rule End-to-End

This document describes rule-based approaches that are evaluated end to end:

1. generate rules from raw documents
2. apply those rules with a downstream rule application strategy
3. measure final QA accuracy, cost, and latency

Unlike the pure rule-generation approaches in [rule_generation.md](/Users/yiminglin/Documents/Codebase/LSF/docs/approach/rule_generation.md), these strategies are not just judged by the quality of the rule set itself. They are judged by the full path from generation to answer quality.

---

## Approach 1 — Agentic Rule Full Data (Codex)

**Code:** `src/baseline/agentic_rule_full_data.py` + wrappers `src/baseline/agentic_rule_full_data_{gpt54,gpt54mini}.py` + runner `src/baseline/run_eval_rule_full_data.py`

### Description

This approach runs one Codex agent session per question over the full `.txt` corpus for a dataset. The agent does **only rule generation**:

- inspect documents on demand
- choose its own working sample
- write Python retrieval rules
- call `verify_accuracy` on the working sample when needed
- stop with a final rule set plus rule-generation metadata

This is intentionally separated from rule application. After generation, the produced rule set can be consumed by any existing application strategy in the repo, such as:

- `src/rule_apply_merge.py`
- `src/default_rule.py`
- `src/rule_apply_individual.py`

So this approach should be evaluated in two phases:
1. **rule generation**: does the agent discover a compact, high-coverage rule set?
2. **rule application**: which downstream application strategy performs best with that rule set?

### Core assumption

Documents in the corpus share strong structural regularities for a fixed question, and those regularities can be captured by a small Python rule set general enough to transfer across the full corpus.

### Inputs

| Item | Source |
|------|--------|
| Question text | passed in prompt |
| Full document corpus (`.txt`) | `data/<dataset>/text/` |
| Ground-truth labels | dataset labels JSON; used only by `verify_accuracy` |
| Helper tools | `list_docs`, `read_doc_txt`, `compute_cost`, `verify_accuracy`, `inspect_rule` |

The agent is not handed the full corpus in-context. It reads docs on demand and manages its own sampling/iteration loop.

### Objectives

| Type | Target |
|------|--------|
| **Hard** | `match_rate >= 0.95` on the agent's chosen working sample, measured by `verify_accuracy` |
| **Soft 1** | Minimize `avg_cost_ratio(r)` |
| **Soft 2** | Maximize per-rule coverage |
| **Soft 3** | Keep `|R|` small |

`verify_accuracy` is the paid tool. Budget: 30 calls per question.

### Interface

```python
def run_rule_gen(
    docs: dict[str, str | Path],              # doc_name -> .txt path
    questions: list[str],                     # exactly one question
    labels_by_doc: dict[str, dict[str, Any]], # GT for verify_accuracy only
    model: str = "gpt54",
    dataset_name: str = "court",
    split_name: str = "all_docs",
    rules_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    run_stem: str = "q01",
) -> dict
```

Wrappers:
- `agentic_rule_full_data_gpt54.py`
- `agentic_rule_full_data_gpt54mini.py`

### Output

The output is a **set of rules**, not baseline QA artifacts.

Rules are written to:

```text
rules/<dataset>/agentic_rule_full_data_<model>/<split>/<qNN>_<question_slug>/
  rule_<name>.py
  ...
```

Rule-generation metadata is written to:

```text
results/<dataset>/agentic_rule_full_data_<model>/<split>/<qNN>_<question_slug>_rule_gen.json
results/<dataset>/agentic_rule_full_data_<model>/<split>/<qNN>_<question_slug>.codex.jsonl
results/<dataset>/agentic_rule_full_data_<model>/<split>/<qNN>_<question_slug>.codex.last.txt
results/<dataset>/agentic_rule_full_data_<model>/<split>/<qNN>_<question_slug>.manifest.json
results/<dataset>/agentic_rule_full_data_<model>/<split>/<qNN>_<question_slug>.verify_accuracy_ledger.json
```

If a downstream rule-application step is run, its outputs should stay under the same strategy root, for example:

```text
results/<dataset>/agentic_rule_full_data_<model>/<split>/rule_apply_merge/
  summary.json
  run_metadata.json
  <question_slug>/<doc_name>.json
  _trace/<question_rule_dir>/<rule_set_slug>_merge.json
```

The final JSON report contains:
- final rule names
- rule directory
- working sample
- number of iterations
- `verify_accuracy` usage/tokens
- Codex token usage for the rule-generation session
- cached/reasoning token counts when available
- Codex log pointers

### Invocation

```bash
python src/baseline/run_eval_rule_full_data.py \
  --dataset court \
  --baseline agentic_rule_full_data_gpt54mini \
  --model gpt54mini \
  --question-slug what_isare_the_court_of_appeals_docket_numbers_for_this_case
```

### Status

Implemented as rule generation only. End-to-end quality is measured only after a separate downstream rule-application step is run.

### Known issue — Python 3.9 type hint bug

Rules that define nested helper functions with `-> str | None` union return annotations crash silently on Python 3.9: the `TypeError` raised at function-definition time is swallowed by the rule's `try/except Exception: return []` wrapper, causing the rule to return zero spans for every document. The fix is to add `from __future__ import annotations` at the top of the rule file (PEP 563 defers annotation evaluation).

Affected rules (identified and patched 2026-05-25): court `q04`, `q05`, `q11`, `q13`. All four rules have been patched and re-run.

### Results

Current checked-in end-to-end runs use `gpt54mini` for rule generation, then `rule_apply_merge` with `gpt54` for answer generation and `gpt54` for judging.

#### Rule Generation

| Dataset | Docs | Queries | Gen model | Avg rules/q | Avg sample match | Avg sample cost | Avg latency/q | Input | Cached input | Output | Reasoning | Path |
|---------|-----:|--------:|-----------|------------:|-----------------:|----------------:|--------------:|------:|-------------:|-------:|----------:|------|
| FinanceBench | 100 | 10 | `gpt54mini` | 1.44 | 0.9259 | 0.003912 | 608.25s | 40,521,866 | 38,570,112 | 420,467 | 237,735 | `results/financebench/agentic_rule_full_data_gpt54mini/all_docs` |
| Court | 294 | 13 | `gpt54mini` | 1.08 | 1 | 0.002883 | 336.79s | 35,095,343 | 33,870,592 | 455,270 | 299,707 | `results/court/agentic_rule_full_data_gpt54mini/all_docs` |
| NOPV | 242 | 12 | `gpt54mini` | 1 | 0.8839 | 0.049122 | 501.61s | 53,754,257 | 51,737,600 | 557,478 | 359,264 | `results/nopv/agentic_rule_full_data_gpt54mini/all_docs` |
| OfficeQA | 200 | 16 | `gpt54mini` | 1.62 | 0.9875 | 0.019666 | 321.99s | 41,828,939 | 39,709,952 | 534,138 | 330,830 | `results/officeqa/agentic_rule_full_data_gpt54mini/all_docs` |

#### Rule Application

| Dataset | Apply strategy | Answer model | Judge model | Docs | Queries | Accuracy | Cost ratio | Latency | Path |
|---------|----------------|--------------|-------------|-----:|--------:|---------:|-----------:|--------:|------|
| FinanceBench | `rule_apply_merge` | `gpt54` | `gpt54` | 100 | 10 | 0.7567 | 0.0134 | 0.84s | `results/financebench/agentic_rule_full_data_gpt54mini/all_docs/rule_apply_merge` |
| Court | `rule_apply_merge` | `gpt54` | `gpt54` | 294 | 13 | 0.6905 | 0.0295 | 0.83s | `results/court/agentic_rule_full_data_gpt54mini/all_docs/rule_apply_merge` |
| NOPV | `rule_apply_merge` | `gpt54` | `gpt54` | 242 | 12 | 0.6791 | 0.0464 | 0.81s | `results/nopv/agentic_rule_full_data_gpt54mini/all_docs/rule_apply_merge` |
| OfficeQA | `rule_apply_merge` | `gpt54` | `gpt54` | 200 | 16 | 0.3738 | 0.1305 | 0.82s | `results/officeqa/agentic_rule_full_data_gpt54mini/all_docs/rule_apply_merge` |
