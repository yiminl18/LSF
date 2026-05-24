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

### Results

Current checked-in results are broken into two phases:

1. rule generation
2. downstream rule application

Only the `court` first-query run has been executed so far for this strategy.

#### Rule Generation

| Dataset | Scope | Question | Gen model | Rules | Match rate | Cost ratio | Latency | Notes |
|---------|-------|----------|-----------|------:|-----------:|-----------:|--------:|-------|
| FinanceBench | — | — | — | — | — | — | — | not run |
| Court | all `294` docs, q1 only | `what_isare_the_court_of_appeals_docket_numbers_for_this_case` | `gpt54mini` | 1 | 1.000 | 0.008398 | 202.21s | `rules/court/agentic_rule_full_data_gpt54mini/all_docs_q1` |
| NOPV | — | — | — | — | — | — | — | not run |
| OfficeQA | — | — | — | — | — | — | — | not run |

Rule-generation token breakdown for the current court run:

| Dataset | Input | Cached input | Output | Reasoning |
|---------|------:|-------------:|-------:|----------:|
| Court | 1,258,334 | 1,198,336 | 29,092 | 22,873 |

#### Rule Application

| Dataset | Apply strategy | Answer model | Scope | Accuracy | Cost ratio | Latency | Notes |
|---------|----------------|--------------|-------|---------:|-----------:|--------:|-------|
| FinanceBench | — | — | — | — | — | — | not run |
| Court | `rule_apply_merge` | `gpt54` | all `294` docs, q1 only | 0.9864 | 0.0332 | 0.76s | `results/court/agentic_rule_full_data_gpt54mini/all_docs_q1/rule_apply_merge` |
| NOPV | — | — | — | — | — | — | not run |
| OfficeQA | — | — | — | — | — | — | not run |

This result is strong because the target field lives in a highly standardized caption/header pattern across most court opinions, so a single header-focused rule transfers well across the corpus.
