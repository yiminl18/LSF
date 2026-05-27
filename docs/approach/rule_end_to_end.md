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

- `src/rule_apply/merge.py`
- `src/rule_apply/default.py`
- `src/rule_apply/individual.py`

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

Rule application uses `rule_apply_merge` with `gpt54` for answer generation and judging throughout.

#### Variant A — Baseline (`gpt54mini` gen, no adaptive sample)

##### Rule Generation

| Dataset | Docs | Q | Avg sample | Avg match | Avg latency | Total input | Total output | Path |
|---------|-----:|--:|-----------:|----------:|------------:|------------:|-------------:|------|
| Court | 294 | 13 | ~7 | 1.000 | 337s | 35,095,343 | 455,270 | `results/court/agentic_rule_full_data_gpt54mini/all_docs` |
| NOPV | 242 | 12 | ~6 | 0.884 | 502s | 53,754,257 | 557,478 | `results/nopv/agentic_rule_full_data_gpt54mini/all_docs` |
| OfficeQA | 200 | 16 | ~4 | 0.988 | 322s | 41,828,939 | 534,138 | `results/officeqa/agentic_rule_full_data_gpt54mini/all_docs` |
| FinanceBench | 100 | 10 | ~5 | 0.926 | 608s | 40,521,866 | 420,467 | `results/financebench/agentic_rule_full_data_gpt54mini/all_docs` |

##### Rule Application

| Dataset | Docs | Q | Accuracy | Cost ratio | Latency | Path |
|---------|-----:|--:|---------:|-----------:|--------:|------|
| Court | 294 | 13 | 0.735 | 0.0295 | 0.83s | `results/court/agentic_rule_full_data_gpt54mini/all_docs/rule_apply_merge` |
| NOPV | 242 | 12 | 0.680 | 0.0464 | 0.81s | `results/nopv/agentic_rule_full_data_gpt54mini/all_docs/rule_apply_merge` |
| OfficeQA | 200 | 16 | 0.374 | 0.1305 | 0.82s | `results/officeqa/agentic_rule_full_data_gpt54mini/all_docs/rule_apply_merge` |
| FinanceBench | 100 | 10 | 0.757 | 0.0134 | 0.84s | `results/financebench/agentic_rule_full_data_gpt54mini/all_docs/rule_apply_merge` |

---

#### Variant B — Adaptive large sample (`gpt54mini` gen, `--adaptive-large-sample`)

##### Rule Generation

| Dataset | Docs | Q | Avg sample | Avg match | Avg latency | Total input | Total output | Path |
|---------|-----:|--:|-----------:|----------:|------------:|------------:|-------------:|------|
| Court | 294 | 13 | 14.5 | 0.969 | 563s | 62,487,212 | 746,101 | `results/court/agentic_rule_full_data_gpt54mini_adaptive/all_docs` |
| NOPV | 242 | 12 | 11.3 | 0.882 | 496s | 60,435,160 | 642,221 | `results/nopv/agentic_rule_full_data_gpt54mini_adaptive/all_docs` |
| OfficeQA | 200 | 16 | 9.3 | 0.876 | 563s | 84,751,470 | 909,176 | `results/officeqa/agentic_rule_full_data_gpt54mini_adaptive/all_docs` |
| FinanceBench | 100 | 12 | 5.2 | 1.000 | 333s | 28,678,167 | 440,890 | `results/financebench/agentic_rule_full_data_gpt54mini_adaptive/all_docs` |

##### Rule Application

| Dataset | Docs | Q | Accuracy | Cost ratio | Latency | Path |
|---------|-----:|--:|---------:|-----------:|--------:|------|
| Court | 294 | 13 | **0.751** | 0.0332 | 0.89s | `results/court/agentic_rule_full_data_gpt54mini_adaptive/all_docs/rule_apply_merge` |
| NOPV | 242 | 12 | **0.822** | 0.1153 | 0.90s | `results/nopv/agentic_rule_full_data_gpt54mini_adaptive/all_docs/rule_apply_merge` |
| OfficeQA | 200 | 16 | **0.386** | 0.1285 | 0.86s | `results/officeqa/agentic_rule_full_data_gpt54mini_adaptive/all_docs/rule_apply_merge` |
| FinanceBench | 100 | 12 | **0.879** | 0.0139 | 0.85s | `results/financebench/agentic_rule_full_data_gpt54mini_adaptive/all_docs/rule_apply_merge` |

---

#### Variant C — Adaptive large sample (`gpt54` gen, `--adaptive-large-sample`)

##### Rule Generation

| Dataset | Docs | Q | Avg sample | Avg match | Avg latency | Total input | Total output | Path |
|---------|-----:|--:|-----------:|----------:|------------:|------------:|-------------:|------|
| Court | 294 | 13 | 17.0 | 0.978 | 545s | 40,650,849 | 410,238 | `results/court/agentic_rule_full_data_gpt54_adaptive/all_docs` |
| NOPV | 242 | 12 | 14.8 | 0.964 | 599s | 48,860,420 | 399,034 | `results/nopv/agentic_rule_full_data_gpt54_adaptive/all_docs` |
| OfficeQA | 200 | 16 | 10.7 | 0.984 | 890s | 101,960,116 | 669,153 | `results/officeqa/agentic_rule_full_data_gpt54_adaptive/all_docs` |
| FinanceBench | 100 | 12 | 6.4 | 0.992 | 477s | 36,531,055 | 328,785 | `results/financebench/agentic_rule_full_data_gpt54_adaptive/all_docs` |

##### Rule Application

| Dataset | Docs | Q | Accuracy | Cost ratio | Latency | Path |
|---------|-----:|--:|---------:|-----------:|--------:|------|
| Court | 294 | 13 | **0.807** | 0.0325 | 0.87s | `results/court/agentic_rule_full_data_gpt54_adaptive/all_docs/rule_apply_merge` |
| NOPV | 242 | 12 | **0.831** | 0.0672 | 0.88s | `results/nopv/agentic_rule_full_data_gpt54_adaptive/all_docs/rule_apply_merge` |
| OfficeQA | 200 | 16 | **0.522** | 0.1288 | 0.84s | `results/officeqa/agentic_rule_full_data_gpt54_adaptive/all_docs/rule_apply_merge` |
| FinanceBench | 100 | 12 | **0.923** | 0.0135 | 0.90s | `results/financebench/agentic_rule_full_data_gpt54_adaptive/all_docs/rule_apply_merge` |

---

#### Summary — Rule Application Accuracy across variants

| Dataset | Baseline (gpt54mini) | Adaptive gpt54mini | Adaptive gpt54 |
|---------|---------------------:|-------------------:|---------------:|
| Court | 0.735 | 0.751 (+1.6pp) | **0.807** (+7.2pp) |
| NOPV | 0.680 | 0.822 (+14.2pp) | **0.831** (+15.1pp) |
| OfficeQA | 0.374 | 0.386 (+1.2pp) | **0.522** (+14.8pp) |
| FinanceBench | 0.757 | 0.879 (+12.2pp) | **0.923** (+16.6pp) |
| **Average** | **0.637** | **0.710** (+7.3pp) | **0.771** (+13.4pp) |
