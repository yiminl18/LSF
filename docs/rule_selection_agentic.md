# Agentic Rule Selection with Claude Opus 4.7

This document specifies a Claude Code–driven agentic approach to rule selection, complementing the algorithmic pipelines documented in `rule_selection_implementation.md` (static-`τ`, auto-tighten) and `rule_selection_pareto_implementation.md` (Pareto frontier). The agentic approach uses an LLM as the outer-loop optimiser: it inspects rules, calls tools to measure their cost / coverage / accuracy, and iterates until a satisfactory rule subset is found.

---

## 1. Goal

Use Claude Code with the Opus 4.7 model as an agent that selects rules for each question, subject to a hard accuracy constraint and soft cost / coverage targets. The agent decides which rules to combine, when to refine or replace one, and when to stop. The algorithmic pipelines already in the codebase remain available; the agent is an alternative outer loop, not a replacement for the underlying primitives.

---

## 2. Problem setting

The corpus is split into two sets:

- **Sampled docs** `D_s`: 10 documents per question (in `data/financebench/sample/single_cluster/random/sample_doc_labels.json`). Used both as the "training" set for rule generation and as the calibration set for selection.
- **Unsampled docs** `D_u`: 50 documents per question (in `data/financebench/sample/single_cluster/random/unsampled_doc_labels.json`). Used as the held-out generalisation set.

The agent's job is to select, from a pre-generated rule pool in `rules/financebench_single_cluster/llm/gpt54/one_shot/<question_slug>_10_llm/`, a small subset `S` that:

1. **Matches the merge accuracy** of the full rule set on `D_s`. Concretely, for every doc `d` in `D_s` that the full rule pool solves correctly (`A*(d) = 1`), the selected subset must also solve correctly: `A(S, d) ≥ A*(d)`. The verifier is the LLM-judge in `src/rule_refinement/eval_judge.py`.
2. **Minimises total cost** on `D_s` — measured as the sum of `avg_cost_ratio(r)` for `r ∈ S`, where the ratio is `retrieved_tokens / total_doc_tokens` per `cost_profile/<slug>.json`.
3. **Keeps each selected rule's coverage high** — measured as `cov(r)` from `eval_individual/<slug>/<r>_eval.json::accuracy`. Higher cov means the rule generalises across documents rather than relying on a single layout.

The selected `S` is then used to extract the minimal context per document at deployment time, which is what reduces token cost on `D_u` while preserving answer accuracy.

So the three optimisation pressures the agent must balance are **cost** (low), **generality / coverage** (high), and **accuracy** (the hard constraint). The first two trade off against each other — broad rules generalise but cost more tokens — and the agent picks the operating point.

---

## 3. Hard vs soft constraints

| Type | Constraint | How the agent checks it |
|---|---|---|
| Hard | `A(S, d) = 1` for every `d ∈ D*_s = {d ∈ D_s : A*(d) = 1}` | Call the `verify_accuracy` tool (LLM-as-judge over merged retrieval) |
| Soft target | Minimise `Σ_{r∈S} avg_cost_ratio(r)` | Call the `compute_cost` tool |
| Soft target | Maximise `min_{r∈S} cov(r)` (or the average / median, depending on emphasis) | Call the `compute_coverage` tool |
| Soft target | Keep `|S|` small (avoid rule bloat) | Track in the agent's working memory |

The agent must not return a final `S` until the hard constraint is satisfied. Soft targets are negotiated against each other; the agent should aim for a defensible trade-off and explain its choice in the final report.

---

## 4. Tools the agent needs

These wrap the primitives already in `src/rule_refinement/`. Each tool is a thin CLI or Python entry point the agent invokes via `Bash`.

### 4.1 `compute_cost(rule_names: list[str])` — free, no LLM

Reads `results/.../cost_profile/<question_slug>.json` (built by `src/rule_refinement/cost_profile.py`) and returns:

```json
{
  "per_rule": {"rule_a": 0.0015, "rule_b": 0.0030, ...},
  "sum_avg_cost_ratio": 0.0080,
  "max_avg_cost_ratio": 0.0030
}
```

Implementation: a one-screen Python script `tools/compute_cost.py` that imports `load_or_compute_cost_profile`.

### 4.2 `compute_coverage(rule_names: list[str])` — free if `eval_individual` is cached

Reads `results/.../eval_individual/<question_slug>/<r>_eval.json::accuracy` for each `r` and returns:

```json
{
  "per_rule": {"rule_a": 0.90, "rule_b": 0.70, ...},
  "min_cov": 0.50,
  "mean_cov": 0.73
}
```

Implementation: `tools/compute_coverage.py` calling `load_or_compute_coverage` from `coverage_check.py`.

### 4.3 `verify_accuracy(rule_names: list[str])` — paid, uses `M_prod`

Runs `rule_apply_merge(rule_names=S, ...)` followed by `eval_judge.judge` for every `d ∈ D_s`, and reports:

```json
{
  "per_doc": [
    {"doc_name": "AMCOR_2019_10K", "correct_S": true,  "correct_full": true},
    {"doc_name": "BOEING_2018_10K", "correct_S": false, "correct_full": true},
    ...
  ],
  "match_rate": 0.90,
  "missed_in_D_star": ["BOEING_2018_10K"]
}
```

The hard-constraint test is `missed_in_D_star == []`. Implementation: `tools/verify_accuracy.py` wrapping `rule_apply_merge` + `judge`.

### 4.4 `list_rules(question_slug: str)` — free

Returns the candidate rule pool with a one-line docstring per rule (parsed from the `rule_<name>` function's docstring). Lets the agent reason about rules semantically before paying for verification.

### 4.5 `inspect_rule(rule_name: str)` — free

Returns the full text of a single rule file. The agent uses this to understand rule structure when deciding which to keep, drop, or combine.

### 4.6 (optional) `refine_rule(rule_name: str, instruction: str)` — paid, uses Opus 4.7

Has Claude rewrite or specialise a rule based on an instruction. Used when the agent decides an existing rule is close but needs adjustment. Outputs a candidate rewrite that must then be re-verified through the other tools.

The first five tools are sufficient for a select-only workflow. Tool 4.6 is the bridge to rule generation / refinement and is optional.

---

## 5. Agent loop

The high-level loop the agent runs per question:

```
1. list_rules(question_slug)               # see the candidate pool
2. compute_cost(all_rules)                 # snapshot of costs
3. compute_coverage(all_rules)             # snapshot of cov values
4. propose initial S based on cov / cost trade-off
5. verify_accuracy(S)
6. while hard constraint not met OR soft targets unsatisfied:
       inspect missed docs, identify which rules might cover them
       inspect_rule on candidates
       update S (add, drop, swap)
       verify_accuracy(S)
7. report final S with cost / coverage / accuracy summary
```

The agent's "thinking" loop is what replaces the hand-coded greedy. Concretely, the agent can:

- Sort the pool by `cov / cost` ratio and walk it (replicating the Pareto algorithm).
- Or look at which docs the full set solves but a candidate `S` misses, then *target* a rule that covers exactly those docs.
- Or notice that two rules have nearly identical coverage and drop the more expensive one (replicating the polish drop-test).
- Or read rule docstrings and notice that several narrow rules could be replaced by one broader rule from the pool — a semantic decision the algorithm cannot make.

The semantic step is the value the agent adds; the algorithmic steps are things the agent can also do but at higher cost than the existing code.

---

## 6. Termination

The agent stops when either:

- **Success**: `verify_accuracy(S)` returns `match_rate = 1.00` on `D*_s`, *and* the agent judges the soft targets to be at a reasonable trade-off. Reasonable can be operationalised as: (a) `min_cov(S) ≥ 0.4`, (b) `sum_avg_cost_ratio(S) ≤ 0.5 × cost_of_full_pool`, (c) `|S| ≤ 10`. If all three soft conditions hold and accuracy matches, return.
- **Stuck**: 5 consecutive iterations with no progress on either accuracy or cost. Return the best `S` seen so far with a "could not improve further" note.
- **Budget exhausted**: Total `verify_accuracy` calls exceed a configured budget (default 30 per question, so roughly 300 LLM-judge invocations).

The agent must produce a final report including the selected rule list, all three metric values, and a one-paragraph rationale.

---

## 7. Implementation outline

### 7.1 Files to add

```
tools/
  compute_cost.py            # 4.1 — wraps cost_profile.load_or_compute_cost_profile
  compute_coverage.py        # 4.2 — wraps coverage_check.load_or_compute_coverage
  verify_accuracy.py         # 4.3 — wraps rule_apply_merge + eval_judge.judge
  list_rules.py              # 4.4 — scans rules/<slug>/, parses docstrings
  inspect_rule.py            # 4.5 — prints rule file contents
  refine_rule.py             # 4.6 — optional, uses Opus 4.7 for rewrite

agent/
  run_agent_select.py        # Outer driver: spawns one Claude Code session per question,
                             # passes the task prompt below, captures the final S to disk.
  task_prompt.md             # The system / task prompt the agent reads on startup
                             # (see §8 below for the template).

results/financebench_single_cluster/llm/gpt54/one_shot/
  selected_rules_agent/<slug>.json  # NEW — final selection + rationale
  agent_trace/<slug>.jsonl          # NEW — per-step tool calls and observations
```

### 7.2 How Claude Code is invoked

`run_agent_select.py` does the equivalent of `task_prompt_rule_gen.py::run(...)` (which already shells out to `claude` for rule generation), but with a selection-focused task prompt:

```python
import subprocess

def run_agent(question, question_slug, model="claude-opus-4-7", cwd=None, budget=30):
    prompt = (Path(__file__).parent / "task_prompt.md").read_text()
    prompt = prompt.format(question=question, question_slug=question_slug, budget=budget)
    res = subprocess.run(
        ["claude", "--model", model, "-p", prompt],
        capture_output=True, text=True, cwd=cwd or Path.cwd(), timeout=3600,
    )
    if res.returncode != 0:
        raise RuntimeError(f"claude exited {res.returncode}: {res.stderr}")
    return res.stdout
```

The tools listed in §7.1 are exposed to the agent via the standard `Bash` tool — the agent invokes them like any other shell command (`python tools/verify_accuracy.py rule_a rule_b ...`). No MCP server is required for the minimal version; an MCP wrapper can be added later if you want richer typed I/O.

### 7.3 Output schema

`selected_rules_agent/<slug>.json` schema:

```json
{
  "question": "...",
  "question_slug": "...",
  "mode": "agentic",
  "model": "claude-opus-4-7",
  "selected_rules": ["rule_a", "rule_b", "rule_c"],
  "selected_avg_cost_ratio_sum": 0.0080,
  "min_cov": 0.60,
  "mean_cov": 0.73,
  "match_rate_on_sampled": 1.00,
  "match_rate_on_unsampled": null,
  "iterations": 7,
  "verify_calls": 5,
  "rationale": "Started with the three highest cov/cost rules. Verification on D_s\nfailed on one doc (BOEING_2018_10K). Inspected the rule pool and added\nrule_x which targets the cover-page table layout used by BOEING. Final\nset preserves accuracy at 1.0 with sum cost 0.0080 (50% of full pool).",
  "trace_path": "results/.../agent_trace/<slug>.jsonl"
}
```

The companion `agent_trace/<slug>.jsonl` records each tool call and its result, one JSON object per line, for post-hoc analysis and debugging.

---

## 8. Task-prompt template (`agent/task_prompt.md`)

The prompt the agent receives at the start of every selection session. Substitution variables are `{question}`, `{question_slug}`, and `{budget}`.

```
You are working inside the LSF project root. Your task is to select a small
subset of rules from a pre-generated rule pool that preserves the merge
accuracy of the full pool on the sampled documents, while keeping total cost
low and per-rule coverage high.

QUESTION   : {question}
SLUG       : {question_slug}
RULE POOL  : rules/financebench_single_cluster/llm/gpt54/one_shot/{question_slug}_10_llm/
SAMPLED    : data/financebench/sample/single_cluster/random/sample_doc_labels.json (10 docs)
COST CACHE : results/financebench_single_cluster/llm/gpt54/one_shot/cost_profile/{question_slug}_10_llm.json
COV CACHE  : results/financebench_single_cluster/llm/gpt54/one_shot/eval_individual/{question_slug}_10_llm/
OUTPUT     : results/financebench_single_cluster/llm/gpt54/one_shot/selected_rules_agent/{question_slug}_10_llm.json

HARD CONSTRAINT (must be satisfied before you finish)
  Merge accuracy of your selected subset S must equal the merge accuracy of
  the full rule pool on every sampled doc the full pool solves correctly.
  Verifier: python tools/verify_accuracy.py --question-slug {question_slug} --rules <rule_a> <rule_b> ...

SOFT TARGETS (negotiate against each other)
  1. Minimise sum(avg_cost_ratio) across S.
  2. Maximise the minimum cov(r) across r in S (broader rules generalise).
  3. Keep |S| small. Prefer one broad rule over three narrow ones.

TOOLS YOU HAVE (invoke via the Bash tool, one per call)
  python tools/list_rules.py --question-slug {question_slug}
  python tools/compute_cost.py --question-slug {question_slug} --rules <names>
  python tools/compute_coverage.py --question-slug {question_slug} --rules <names>
  python tools/inspect_rule.py --question-slug {question_slug} --rule <name>
  python tools/verify_accuracy.py --question-slug {question_slug} --rules <names>

LOOP
  1. List rules and read their docstrings.
  2. Snapshot cost and coverage for the whole pool.
  3. Propose an initial S using a sensible heuristic — cov/cost descending is a
     good starting point.
  4. Verify accuracy. If misses exist, inspect the missed docs' likely rules
     and add or swap.
  5. Once accuracy matches, try drop-tests: can you remove the most expensive
     rule and still pass?
  6. Stop when accuracy matches AND your soft targets feel reasonable, or when
     you have used {budget} verify_accuracy calls.

OUTPUT
  When done, write the final JSON to the OUTPUT path with this schema:
    {{ "selected_rules": [...], "selected_avg_cost_ratio_sum": ...,
       "min_cov": ..., "mean_cov": ..., "match_rate_on_sampled": ...,
       "iterations": ..., "verify_calls": ..., "rationale": "..." }}
  Then print a one-line summary to stdout.

GUIDELINES
  - Prefer broad rules with cov >= 0.5 over narrow rules with cov < 0.3.
  - A rule that "wins" on cost but covers only one doc is almost always
    overfit; treat it with suspicion.
  - Read rule docstrings before adding a rule. Rules whose names contain
    page numbers or specific dates are often layout-specific and overfit.
  - Do not refuse to finish; if you cannot satisfy the hard constraint within
    {budget} verify calls, return the best S you have and explain why.
```

---

## 9. Comparison with the algorithmic pipelines

| Aspect | Static-`τ` / Auto-tighten / Pareto (algorithmic) | Agentic (this doc) |
|---|---|---|
| Outer loop | Hand-coded greedy in `select_rules*.py` | Claude Opus 4.7 inside Claude Code |
| Determinism | Deterministic given the same inputs | Non-deterministic |
| Cost per question | ~30–80 `M_prod` judge calls | ~10–30 `M_prod` judge calls + Opus 4.7 reasoning tokens |
| Semantic reasoning | None — works on cov / cost numbers | Reads rule docstrings; can spot layout-specific rules; can explain choices |
| Rule refinement | Cannot create or modify rules | Can with the optional `refine_rule` tool |
| Output shape | Single `S` (or frontier for Pareto) | Single `S` + natural-language rationale + trace |
| Verifier | `eval_judge.judge` | Same `eval_judge.judge`, invoked via tool |
| Best use case | Production runs, repeatable, cheap | Hard questions where the algorithm overfits; exploratory analysis; rule refinement |

A practical pattern is to run the Pareto pipeline first to get a baseline frontier, then run the agent on the few questions where the Pareto-selected set still underperforms on `D_u` (visible from the overfit-report analysis). The agent's semantic reasoning over rule docstrings is exactly what's needed to spot the misses that pure cov/cost optimisation cannot.

---

## 10. Operational notes

- **Concurrency.** Run one Claude Code session per question. They are independent and can be parallelised across questions, subject to model rate limits.
- **Reproducibility.** Even though Opus 4.7 is non-deterministic, the trace at `agent_trace/<slug>.jsonl` records every tool call. Replaying the same tool sequence is deterministic; only the agent's *choices* are not.
- **Cost control.** Cap `verify_accuracy` calls per session (default 30). Cap total Opus 4.7 tokens via the `--max-tokens` flag on `claude`. Persist the cost-profile and coverage caches so the cheap tools never hit the LLM.
- **Failure mode.** If the agent finishes without satisfying the hard constraint, the output JSON should still be written with `match_rate_on_sampled < 1.0` and the rationale explaining why. Downstream pipelines can detect and either fall back to the algorithmic selection or re-run the agent with a larger budget.
- **Comparison runs.** Persist agentic results to `selected_rules_agent/` separately from `selected_rules/`, `selected_rules_auto/`, `selected_rules_pareto/` so that all four selection strategies (static, auto-tighten, Pareto, agentic) can be compared per question on both `D_s` and `D_u`.

---

## 11. Open questions

- **How much does the agent add over the Pareto frontier?** The Pareto algorithm already does cost/coverage-aware selection. The agent's marginal value is semantic reasoning about rule docstrings; this is hardest to quantify without an ablation. Recommend a small evaluation: run both on the 10 sample questions, compare `match_rate_on_unsampled` per question.
- **Should the agent be allowed to call `refine_rule`?** Refinement turns rule selection into rule generation, which is a more powerful but more expensive workflow. Start without it; enable it for specific questions where the existing rule pool is clearly insufficient (visible from low Pareto frontier ceiling).
- **Prompt iteration.** The task prompt in §8 is a starting template. Track which prompt variants improve `match_rate_on_unsampled` and update accordingly. Treat the prompt itself as a hyperparameter of the agentic pipeline.

---

## 12. Implementation status — linkage to existing codebase

This section maps every tool and component in this spec to the concrete code now present in the repo, plus the primitives in `src/rule_refinement/` that the tools wrap.

### 12.1 Tools (`tools/`)

All implemented as standalone Python CLIs under `tools/`. Each is a thin wrapper over an existing primitive in `src/rule_refinement/` — no new business logic, only argument parsing + result formatting.

| Spec §4 | File | Wraps | LLM cost | Notes |
|---|---|---|---|---|
| §4.1 `compute_cost` | `tools/compute_cost.py` | `src/rule_refinement/cost_profile.py::load_or_compute_cost_profile` | Free | Builds + caches profile in `cost_profile/<slug>.json` on first call; subsequent calls free. |
| §4.2 `compute_coverage` | `tools/compute_coverage.py` | `src/rule_refinement/coverage_check.py::load_or_compute_coverage` | Free | Reads `eval_individual/<slug>/<rule>_eval.json::accuracy`; falls back to `0.0` if missing. |
| §4.3 `verify_accuracy` | `tools/verify_accuracy.py` | `src/rule_apply_merge.py::rule_apply_merge` + `src/rule_refinement/eval_judge.py::judge` + `src/rule_refinement/baseline_targets.py::load_target_docs` | 2 gpt54 calls × 10 sampled docs ≈ 20 calls/invocation | Computes `D*` from `eval_merge/<slug>_sampled.json`, runs merge + judge for the subset, reports `match_rate` and `missed_in_D_star`. |
| §4.4 `list_rules` | `tools/list_rules.py` | `ast.parse` on each `rule_*.py` to extract function docstrings | Free | Scans `rules/.../<slug>/`, returns one-line docstrings via Python AST. |
| §4.5 `inspect_rule` | `tools/inspect_rule.py` | direct file read | Free | Prints raw `rule_<name>.py` source. |
| §4.6 `refine_rule` (optional) | not yet implemented | (would wrap `src/rule_gen_agent_claude.py`) | Paid (Opus 4.7) | Deferred per spec — select-only workflow doesn't need it. |

Shared path constants live in `tools/_paths.py` so the tools agree on where rules, labels, and caches are.

### 12.2 Agent driver (`agent/`)

| Spec §7.2 | File | Notes |
|---|---|---|
| `run_agent_select.py` | `agent/run_agent_select.py` | Mirrors `src/rule_gen_agent_claude.py::run`'s `subprocess.run(["claude", "--model", ..., "-p", prompt])` pattern. CLI flags: `--slug`, `--budget`, `--model`, `--dry-run`, `--timeout`. |
| `task_prompt.md` | `agent/task_prompt.md` | Filled with `{question}`, `{question_slug}`, `{budget}`, `{model}`, `{output_path}`, `{trace_path}` placeholders. The body matches the spec §8 template, expanded with concrete tool invocation lines and the AGENTIC_SELECTION_DONE summary-line format the driver parses. |

Model aliases (`opus`, `opus47`, `sonnet`, `haiku`) match the convention already used in `src/rule_gen_agent_claude.py`.

### 12.3 Output layout

Implemented exactly as §7.1 specifies. Output paths come from `tools/_paths.py`:

```
results/financebench_single_cluster/llm/gpt54/one_shot/
  selected_rules_agent/<slug>.json   # written by the agent itself (§7.3 schema)
  agent_trace/<slug>.jsonl           # written by the agent as it tool-calls
  selector_run_agent/<slug>/...      # intermediate merge predictions from verify_accuracy
```

### 12.4 Sanity checks performed

After implementation:
1. `python3 tools/list_rules.py --question-slug what_is_the_registrants_telephone_number_10_llm` returned 78 rules with docstrings ✓
2. `python3 tools/inspect_rule.py --rule rule_address_and_telephone_number_of_registrant ...` returned full source ✓
3. `python3 tools/compute_cost.py --rules <a> <b>` returned `sum_avg_cost_ratio=0.000209` from the cached profile ✓
4. `python3 tools/compute_coverage.py --rules <a> <b>` returned `mean_cov=0.05` from `eval_individual/` cache ✓
5. `python3 agent/run_agent_select.py --slug ... --dry-run` built a 6223-character prompt without errors ✓

`verify_accuracy.py` was not smoke-tested with real LLM calls but its imports succeed and its logic is a straightforward composition of two already-tested primitives.

### 12.5 What the agent actually sees

Translating one cell of the task prompt to a concrete tool invocation. When the agent runs:

```bash
python tools/verify_accuracy.py \
    --question-slug what_is_the_registrants_telephone_number_10_llm \
    --question "What is the registrant's telephone number?" \
    --rules rule_address_and_telephone_number_of_registrant rule_page1_phone_number_pattern
```

…it invokes (under the hood):

```python
# tools/verify_accuracy.py
from rule_apply_merge import rule_apply_merge
from rule_refinement.eval_judge import judge
from rule_refinement.baseline_targets import load_target_docs

# 1. Load D* from existing eval_merge artifact (no LLM)
D_star = load_target_docs(eval_merge_path)

# 2. For each sampled doc, apply rules + run gpt54 QA + run gpt54 judge
for dn in doc_names:
    pred = rule_apply_merge(document=..., rule_names=["rule_..."]).predicted_answer  # 1 gpt54 call
    ok, _, _ = judge(question, gt, pred, model_name="gpt54")                          # 1 gpt54 call
    ...

# 3. Compare correct_S vs D* → missed_in_D_star
```

So every tool call from the agent's perspective is a single shell command; under the hood it composes the existing tested primitives. No new LLM prompt engineering — all of that was done in `eval_judge.py` and `rule_apply_merge.py` already.

### 12.6 Running the agent end-to-end

```bash
# All 10 questions:
python agent/run_agent_select.py --model opus47 --budget 20

# One question:
python agent/run_agent_select.py \
    --slug what_is_the_registrants_telephone_number_10_llm \
    --model opus47 --budget 15

# Just print the expanded prompt without spending tokens:
python agent/run_agent_select.py --slug <slug> --dry-run
```

The driver writes a final summary to stdout. Per-question outputs land in `selected_rules_agent/<slug>.json` (written by the agent itself, matching §7.3 schema).
