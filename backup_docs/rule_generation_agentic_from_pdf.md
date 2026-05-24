# Agentic Rule Generation from PDFs

This document specifies a Claude Code–driven agentic approach to **rule generation from scratch**. The agent receives **the question, ground-truth labels, and the reconstructed JSON representation of each sampled PDF** — no pre-existing rules of any kind. Its task is to **invent the rule set** by inspecting the JSONs (the same structure rules consume at runtime) and writing new Python rule functions, iterating against the constraints until the merge accuracy target is met.

> **Note on title:** JSON is the recommended representation of the PDF data. The spec is named `*_from_pdf.md` because the upstream source is PDFs, but the agent never reads PDFs directly — it operates on `data/financebench/processing/<stem>_reconstructed.json`. PDF reconstruction is a one-time offline pipeline run before the agent starts.

It is the generation-side counterpart of `docs/rule_selection_agentic.md`. The two specs differ in input and output:

| | `rule_selection_agentic.md` | **this spec** |
|---|------------------------------|----------------|
| Input | Pre-generated rule pool + reconstructed JSON | **Only reconstructed JSON + question + labels (no rules at all)** |
| Agent's job | Pick a subset from the existing pool | **Write new rule code from scratch** |
| Output | A list of rule names already in the pool | **New `.py` files containing rule functions** |

The driver mirrors `agent/run_agent_select.py`'s pattern: spawn one Claude Opus 4.7 session per question, expose a fixed set of tools the agent invokes via Bash, capture the result + trace per session. The differences are the **inputs** (raw JSON docs instead of a rule pool), the **tools** (`list_docs` + `read_doc_json` + `write_rule` added to the five shared verifiers), and the **output** (rule `.py` files plus a session summary JSON, instead of just a selection JSON).

---

## 1. Goal

Use Claude Code with Opus 4.7 as an agent that **generates a minimal set of Python span-retrieval rules from scratch** for each question. The agent's session starts with **no rules of any kind** — its only inputs are the reconstructed JSON for each sampled doc, the question text, and the ground-truth labels. It must inspect the JSON spans, identify where the answer lives, and write Python rule functions that retrieve that content. The hard merge-accuracy constraint and the soft cost / rule-count targets define when it can stop.

The output rules follow the same signature and downstream-compatibility contract as the existing LLM-coarse and agent-raw pipelines: a Python file per rule, each exporting a `def rule_<name>(doc: dict) -> list[dict]` that retrieves spans matching some pattern. This means the generated rules plug into the existing `rule_apply_merge` / `eval_judge` / Pareto-selection / agentic-selection pipelines unchanged.

---

## 1.5. Inputs and outputs (explicit)

### What the agent has at session start

| Item | Source |
|------|--------|
| The question text | `data/financebench/sample_queries.txt` (one line, passed to the agent in its prompt) |
| The reconstructed JSON for each sampled doc (10 files) | `data/financebench/processing/<stem>_reconstructed.json` (offline-built; cached) |
| Ground-truth labels for the sampled docs | `data/financebench/sample/single_cluster/random/sample_doc_labels.json` (read by `verify_accuracy`, not directly by the agent) |
| The hard constraint and soft targets | Spelled out in the task prompt |
| The set of tools | JSON inspection + rule authoring + rule testing — see §4 |

### What the agent does NOT have at session start

| Item | Why excluded |
|------|--------------|
| Any rule files | The agent generates them. The output directory starts empty. |
| Any pre-built rule pool | This is the key distinction from `rule_selection_agentic.md`. |
| Raw PDFs | The agent never opens PDFs. Reconstruction has already happened offline; the agent reads only the cached JSON. |
| Coverage / cost statistics | No `cov(r)` to look up — these only exist after a rule is written and tested. |

### What the agent produces

| Output | Path |
|--------|------|
| Generated rule files (one per rule) | `rules/financebench/lsf/single_cluster/agent/opus47/agentic{,_fps}/raw/<slug>_10_agentic{,_fps}/rule_<name>.py` |
| Session summary | `results/financebench/lsf/single_cluster/agent/opus47/agentic{,_fps}/raw/selected_rules_gen/<slug>.json` |
| Per-tool-call trace | `results/financebench/lsf/single_cluster/agent/opus47/agentic{,_fps}/raw/agent_trace/<slug>.jsonl` |

The output rule files are the only artifact downstream pipelines need. They drop straight into existing infrastructure (`rule_apply_merge`, Pareto selection, agentic selection, etc.).

---

## 2. Problem setting

The corpus is the existing **reconstructed-JSON directory** at `data/financebench/processing/`. Concretely:

- **Sampled docs** `D_s`: 10 reconstructed JSONs (calibration set, with ground-truth answers in a labels file).
- **Unsampled docs** `D_u`: 50 reconstructed JSONs (held-out generalization set, also with ground-truth answers).
- **The PDF→JSON reconstruction pipeline** has already been run offline; the agent never opens a PDF. The reason the spec is called `*_from_pdf.md` is the upstream pipeline that produced these JSONs.

The agent's job is to author a small rule set `R ⊆ rules generated this session` such that the union of their retrievals supports correct answers on `D_s`.

The rule pool is **created**, not chosen from. The agent starts with the empty set and writes new Python rule files into a per-question output folder as it works.

---

## 3. Hard vs soft constraints

**These match `docs/rule_selection_agentic.md` §3 exactly.** The only difference is the underlying rule set: in selection it's a subset `S` chosen from a pre-existing pool; here it's a set `R` the agent has generated so far in this session. The formulas, the targets, and the tools that verify them are identical.

| Type | Constraint | How the agent checks it |
|---|---|---|
| Hard | `A(R, d) = 1` for every `d ∈ D*_s` | Call the `verify_accuracy` tool (LLM-as-judge over merged retrieval, gpt54) |
| Soft target | Minimise `Σ_{r∈R} avg_cost_ratio(r)` | Call the `compute_cost` tool |
| Soft target | Maximise `min_{r∈R} cov(r)` (or mean / median, depending on emphasis) | Call the `compute_coverage` tool |
| Soft target | Keep `|R|` small (avoid rule bloat) | Track in the agent's working memory |

`A(R, d)` denotes the merge accuracy of rule set `R` on doc `d` — apply every rule in `R`, union the retrieved spans, send the concatenated text + question to gpt54, judge with gpt54. `D*_s` is the set of docs answerable from the corpus; in the generation setting where no pre-existing pool defines a ceiling, **`D*_s = D_s` (all 10 sampled docs)** — the agent is expected to cover every sampled doc that has a ground-truth label. `cov(r)` is the fraction of sampled docs that rule `r` alone retrieves the correct answer for.

The agent must not return a final `R` until the hard constraint is satisfied. Soft targets are negotiated against each other; the agent should aim for a defensible trade-off and explain its choice in the final report.

---

## 4. Tools the agent needs

The agent uses the **same five constraint-verification tools as `docs/rule_selection_agentic.md` §4**, plus a small set of PDF inspection tools and a `write_rule` tool to author new rule files. The constraint-verification tools (compute_cost, compute_coverage, verify_accuracy) operate on whatever rule names the agent passes — selection-agentic passes names from a pre-built pool; generation-from-PDF passes names of rules it has just written this session.

### 4.1 `compute_cost(rule_names)` — free, no LLM

**Same tool as selection §4.1.** Computes `avg_cost_ratio(r)` for each named rule across the sampled docs by applying the rule, tokenising the retrieved text, and dividing by total doc tokens. Returns per-rule cost + sum + max. No LLM.

```bash
python tools/compute_cost.py --question-slug <slug> --rules <r1> <r2> ...
```

For generation, the agent passes names of rules it has already written this session (`compute_cost --rules rule_a rule_b`); for selection, names from the pre-existing pool. Both invocations are identical — the tool just reads whichever rule `.py` files exist.

### 4.2 `compute_coverage(rule_names)` — free, no LLM

**Same tool as selection §4.2.** Returns `cov(r)` for each named rule (fraction of sampled docs where the rule alone retrieves the GT answer correctly). For pre-existing rules this looks up `eval_individual/<slug>/<r>_eval.json`; for newly-generated rules with no cached eval, computes via the substring proxy and (optionally) the LLM judge.

```bash
python tools/compute_coverage.py --question-slug <slug> --rules <r1> <r2> ...
```

### 4.3 `verify_accuracy(rule_names)` — paid (gpt54 QA + judge, ~20 calls per invocation)

**Same tool as selection §4.3.** The hard-constraint verifier. Applies the rule union to each sampled doc, runs gpt54 QA on the merged retrieved text, runs gpt54 judge against ground truth, reports per-doc verdicts + overall `match_rate`. The agent cannot finalize until `match_rate = 1.0` on `D*_s`.

```bash
python tools/verify_accuracy.py --question-slug <slug> \
                                 --question "..." \
                                 --rules <r1> <r2> ...
```

This is the single source of truth for the hard constraint, used identically by both selection and generation agents.

### 4.4 `list_rules(question_slug)` — free

**Same tool as selection §4.4.** Returns the rules currently in the question's rule folder, with one-line docstrings. For selection-agentic the folder is pre-populated; for generation-from-PDF it starts empty and grows as the agent calls `write_rule`. Either way, the tool's behaviour is identical.

```bash
python tools/list_rules.py --question-slug <slug>
```

### 4.5 `inspect_rule(rule_name)` — free

**Same tool as selection §4.5.** Returns the full source of one rule file. Used after `list_rules` to read a rule's code — typically to verify what it does, or to use it as a template for writing a related rule.

```bash
python tools/inspect_rule.py --question-slug <slug> --rule <name>
```

---

### Additional tools needed by generation (not in selection-agentic)

The five tools above are shared. Generation adds these because the agent needs to inspect documents (rather than browse a pre-built rule pool) and must persist new rule files.

### 4.6 `list_docs(labels-file)` — free

Returns the list of reconstructed-JSON paths for every doc in the given labels file (the agent's "sampled set"), with per-doc span count and page count.

```bash
python tools/list_docs.py --labels-file data/financebench/sample/single_cluster/random/sample_doc_labels.json
```

### 4.7 `read_doc_json(stem, page|pages|filter)` — free

Loads one reconstructed JSON and returns a focused view of its spans — the same field set the rule's `doc: dict` receives at runtime (`text`, `page_no`, `size`, `bold`, `label`, `structure.{level, path_text, depth}`). Filter by single page, page range, or case-insensitive substring; cap with `--max-spans`.

```bash
python tools/read_doc_json.py --doc AMCOR_2019_10K --page 1
python tools/read_doc_json.py --doc AMCOR_2019_10K --pages 1-3 --filter "exact name" --max-spans 80
```

### 4.8 `write_rule(question-slug, name, code)` — free

Persists a new rule to `<rules-dir>/<question-slug>/<rule_name>.py`. Validates the `def rule_<name>(doc: dict) -> list[dict]` signature via AST and runs a smoke import to catch syntax / NameError issues before returning.

```bash
python tools/write_rule.py --question-slug <slug> --name rule_<name> \
    --code-file /tmp/proposed.py --rules-dir <rules-dir>
```

After this call, the rule is visible to `list_rules`, `inspect_rule`, `compute_cost`, and `verify_accuracy` — every shared verification tool operates on it identically to how it operates on a pre-existing pool rule.

---

### Tool tiers (cost discipline)

| Tier | Tools |
|------|-------|
| Free, no LLM | `compute_cost`, `list_rules`, `inspect_rule`, `list_docs`, `read_doc_json`, `write_rule` |
| Soft-target signal | `compute_coverage` returns 0.0 for newly-written rules (no cache). For real `cov(r)` on a fresh rule, call `verify_accuracy --rules <single_rule>` — that match_rate **is** `cov(r)`. |
| Paid, expensive | `verify_accuracy` with `--d-star-mode all_labeled` (~20 gpt54 calls per invocation; budget-capped, default 30/Q) |

`verify_accuracy` is the **only** tool capped by the per-question budget, matching selection-agentic's policy.

---

## 5. Agent loop

Mirrors `docs/rule_selection_agentic.md` §5 — same constraint/objective formulation, same verification tools. The only differences are (a) the rule set starts empty and grows via `write_rule`, (b) the inspection step uses `list_docs` + `read_doc_json` over the reconstructed JSON corpus instead of `list_rules`/`inspect_rule` over a pre-built rule pool. After a rule is written, the constraint-checking tools (`compute_cost`, `verify_accuracy`) work identically to the selection case.

High-level loop per question:

```
1. list_docs(labels_file)                            # see the 10 reconstructed JSONs
2. read_doc_json(stem_a, page=1)                     # inspect representative docs
   read_doc_json(stem_b, pages=1-3, filter="...")    # focus on the relevant region

3. Propose an initial rule based on observed patterns
4. write_rule(<name>, <code>)                        # rule now persisted

5. compute_cost(rules=[<name>])                      # free; per-rule cost
   verify_accuracy(rules=[<name>])                   # paid; gives match_rate = cov(<name>)
                                                     # (counts against the verify budget)

6. While hard constraint not met OR soft targets unsatisfied:
       verify_accuracy(rules=R)                      # paid; the hard-constraint check
       inspect missed docs (read_doc_json on docs
           where match_rate < 1)
       Decide: write a new rule, modify an existing rule,
           or drop a rule (write_rule --overwrite / Write)
       Re-check soft targets via compute_cost

7. Report final R with cost / coverage / accuracy summary
```

The agent's freedom — same as the selection-agentic version, with two additions:

- **Cross-doc reasoning**: read 2–3 JSONs before writing a rule, so the rule generalises rather than overfits one filer's template.
- **Adaptive specificity**: start broad, specialise only for docs not yet covered.
- **Explanation**: each rule's docstring is written by the agent, captured in `inspect_rule`'s output.

Cost-controlled inner / outer loop, matching selection-agentic:

- **Cheap inner loop**: `compute_cost`, `list_rules`, `list_docs`, `read_doc_json`, `write_rule`. The agent iterates here freely.
- **Expensive outer check**: `verify_accuracy` (gpt54 QA + judge over the merged retrieval). Used both to check the hard constraint on the full set and to derive `cov(r)` on a single new rule. Budget-capped at 30 calls/Q (matching selection §4.3).

---

## 6. Termination

**Same as `docs/rule_selection_agentic.md` §6.** The agent stops when either:

- **Success**: `verify_accuracy(R)` returns `match_rate = 1.00` on `D*_s`, *and* the agent judges the soft targets to be at a reasonable trade-off. Reasonable can be operationalised as: (a) `min_cov(R) ≥ 0.4`, (b) `sum_avg_cost_ratio(R) ≤ 0.5 × cost_of_naive_full_retrieval` (i.e. retrieved tokens ≤ 50% of the doc), (c) `|R| ≤ 10`. If all three soft conditions hold and accuracy matches, return.
- **Stuck**: 5 consecutive iterations with no progress on either accuracy or cost. Return the best `R` seen so far with a "could not improve further" note.
- **Budget exhausted**: total `verify_accuracy` calls exceed the configured budget (default 30 per question — same as selection-agentic).

The agent must produce a final report including the rule list, all three metric values, and a one-paragraph rationale.

---

## 7. Implementation outline

### 7.1 Files to add

```
tools/
  # shared with rule_selection_agentic (§4.1–§4.5 of that spec)
  compute_cost.py          # avg_cost_ratio(r) per rule — free
  compute_coverage.py      # cov(r) per rule — free if cached, 0.0 for newly-written rules
  verify_accuracy.py       # the hard-constraint verifier — paid (gpt54 QA + judge)
                           # now accepts --d-star-mode {file, all_labeled}:
                           # 'all_labeled' is used by generation (D* = all sampled docs)
  list_rules.py            # list rules currently in the question's rule folder
  inspect_rule.py          # read one rule's source

  # added by generation (JSON inspection + rule authoring)
  list_docs.py             # list reconstructed-JSON paths from a labels file
  read_doc_json.py         # show a focused view of one reconstructed JSON's spans
  write_rule.py            # validate signature + persist a new rule .py file

agent/
  run_agent_gen.py                # outer driver: spawns one Claude session per question
  task_prompt_gen.md              # task prompt template

# Task 1 (random sample) outputs:
rules/financebench/lsf/single_cluster/agent/opus47/agentic/raw/
  <slug>_10_agentic/rule_<name>.py            # generated rules
results/financebench/lsf/single_cluster/agent/opus47/agentic/raw/
  selected_rules_gen/<slug>.json              # session summary
  agent_trace/<slug>.jsonl                    # per-step trace
  eval_merge_sampled/<slug>_sampled.json      # downstream eval on the 10 sampled docs
  eval_merge_unsampled/<slug>_unsampled.json  # downstream eval on the 50 unsampled docs

# Task 2 (FPS sample) outputs: same shape under .../agentic_fps/raw/ with slugs ending in _10_agentic_fps.
```

### 7.2 How Claude Code is invoked

Identical pattern to `agent/run_agent_select.py`:

```python
result = subprocess.run(
    ["claude", "--model", "claude-opus-4-7",
     "--output-format", "json",
     "--dangerously-skip-permissions",
     "-p", prompt],
    capture_output=True, text=True, cwd=project_root, timeout=3600,
)
```

The driver fills in `{question}`, `{question_slug}`, `{labels_file}`, `{processing_dir}`, `{rules_dir}`, `{output_path}`, `{trace_path}`, `{cost_cache_dir}`, `{eval_individual_dir}`, `{selector_run_dir}`, `{budget}`, and `{model}` placeholders in `agent/task_prompt_gen.md`, and captures stdout for the AGENTIC_GEN_DONE summary line.

### 7.3 Output schema

`selected_rules_gen/<slug>.json` — same field names as `rule_selection_agentic.md` §7.3, so downstream summary scripts can read both:

```json
{
  "question":                     "...",
  "question_slug":                "...",
  "mode":                         "agentic_gen_from_pdf",
  "model":                        "claude-opus-4-7",
  "selected_rules":               ["rule_a", "rule_b", "rule_c"],
  "selected_avg_cost_ratio_sum":  0.032,
  "min_cov":                      0.6,
  "mean_cov":                     0.78,
  "match_rate_on_sampled":        1.0,
  "iterations":                   7,
  "verify_calls":                 4,
  "tool_llm_calls":               X,
  "tool_input_tokens":            X,
  "tool_output_tokens":           X,
  "latency_seconds":              X,
  "rationale":                    "After inspecting AMCOR p1 + p2 (rendered with read_pdf_vision), wrote a broad rule for cover-page bold spans. verify_accuracy failed on EBAY; inspected EBAY p1 — added a sibling rule for the alternative right-column layout. Final 2 rules cover all sampled docs at sum cost 0.032."
}
```

The companion `agent_trace/<slug>.jsonl` records each tool call and its result.

---

## 8. Task-prompt template (`agent/task_prompt_gen.md`)

Mirrors `agent/task_prompt.md` (the selection prompt) — same hard / soft constraint wording, same verification tools — with the rule-pool browsing block replaced by a JSON-inspection block (`list_docs` + `read_doc_json`) and a `write_rule` step.

```
You are working inside the LSF project root. Your task is to GENERATE a small
set of Python span-retrieval rules from scratch for the following question,
by inspecting the sampled PDF documents and writing new rule files.

QUESTION   : {question}
SLUG       : {question_slug}
LABELS     : {labels_file}
PROCESSING : {processing_dir}        (reconstructed JSON, one per doc)
RULES DIR  : {rules_dir}             (write rules to <rules_dir>/{question_slug}/)
OUTPUT     : {output_path}           (your final session-summary JSON)
TRACE      : {trace_path}

HARD CONSTRAINT (must be satisfied before you finish)
  For every sampled doc d, match_rate(R, d) = 1.
  Equivalently:
    python tools/verify_accuracy.py --question-slug {question_slug} \
        --question "{question}" --rules <r1> <r2> ... \
        --rules-dir {rules_dir} --labels-file {labels_file} \
        --d-star-mode all_labeled
  must return `missed_in_D_star: []`.

SOFT TARGETS (negotiate against each other)
  1. Minimise sum(avg_cost_ratio) across R         — tool: compute_cost.py
  2. Maximise min cov(r) across r in R             — call verify_accuracy on a
                                                     single rule; the match_rate
                                                     IS cov(r). compute_coverage
                                                     returns 0.0 for fresh rules.
  3. Keep |R| small. Prefer one broad rule over three narrow ones.

TOOLS YOU HAVE (invoke via the Bash tool)

  # Shared verification (same five tools as rule_selection_agentic)
  python tools/compute_cost.py     --question-slug {question_slug} --rules <r1> ...
  python tools/compute_coverage.py --question-slug {question_slug} --rules <r1> ...
  python tools/verify_accuracy.py  --question-slug {question_slug} \
        --question "{question}" --rules <r1> ... --d-star-mode all_labeled
  python tools/list_rules.py       --question-slug {question_slug}
  python tools/inspect_rule.py     --question-slug {question_slug} --rule <name>

  # JSON inspection (new for generation)
  python tools/list_docs.py        --labels-file {labels_file}
  python tools/read_doc_json.py    --doc <stem> --page <N>           # or --pages, --filter

  # Rule authoring (new for generation)
  python tools/write_rule.py       --question-slug {question_slug} \
        --name rule_<name> --code-file /tmp/<name>.py --rules-dir {rules_dir}

BUDGET: at most {budget} verify_accuracy calls per question (default 30).

LOOP
  1. list_docs to see the 10 reconstructed JSONs.
  2. read_doc_json on 2-3 representative docs to understand structure.
  3. Author rule_<name>.py via write_rule.
  4. compute_cost + verify_accuracy (single-rule) for the first rule.
  5. While the union misses some docs: inspect missed docs via read_doc_json,
     add a sibling rule or revise existing rule, re-verify.
  6. Stop when match_rate=1.0 on all 10 sampled docs AND soft targets feel
     reasonable, or budget exhausted.

OUTPUT — session summary JSON at {output_path} with schema:
  question, question_slug, mode="agentic_gen", model, selected_rules,
  selected_avg_cost_ratio_sum, min_cov, mean_cov, match_rate_on_sampled,
  iterations, verify_calls, tool_llm_calls, tool_input_tokens,
  tool_output_tokens, latency_seconds, rationale.

Append per-call trace to {trace_path}. Then print to stdout:
  AGENTIC_GEN_DONE slug={question_slug} n_rules=N sum_cost=F \
                   min_cov=F match_rate=F tool_calls=N latency_s=F
```

---

## 9. Comparison with the existing rule-gen pipelines

| Aspect | `rule_gen_llm_coarse` | `rule_gen_agent_claude` | **`run_agent_gen`** (this) |
|---|---|---|---|
| Document input | Reconstructed JSON spans | Reconstructed JSON spans | Reconstructed JSON spans |
| Outer loop | Single prompt, one shot | Claude inside Claude Code, JSON tools | Claude inside Claude Code, **same 5 verification tools as `rule_selection_agentic` + `list_docs`/`read_doc_json`/`write_rule`** |
| Hard-constraint check | Indirect (downstream eval only) | Indirect | **Direct, every iteration:** `verify_accuracy --d-star-mode all_labeled` |
| Determinism | Deterministic at temp=0 | Non-deterministic | Non-deterministic |
| Output | Rule pool (~100 rules) | Rule pool (~5-20 rules with rationale) | Rule pool (~2-5 rules, generation grounded in verifier feedback) |
| When to prefer | Mass rule production, cheap | Hard questions where mass rules underperform | When you want the agent to *prove* per-doc accuracy as it writes |

The new variant differs from `rule_gen_agent_claude` in **how it iterates**: it shares the same five constraint-verification tools as `rule_selection_agentic`, so the loop is structurally identical to selection — only the inspection block is different (JSON inspection + `write_rule` instead of pool browsing).

---

## 10. Operational notes

- **Concurrency.** One Claude Code session per question; sessions are independent and parallelizable subject to model rate limits.
- **Reproducibility.** Even though Opus is non-deterministic, the trace at `agent_trace/<slug>.jsonl` records every tool call. Replaying the same tool sequence is deterministic; only the agent's choices are not.
- **Cost control.** Cap `verify_accuracy` calls per session (default 30, matching `rule_selection_agentic.md` §4.3). `verify_accuracy` is the only paid tool (~20 gpt54 calls per invocation); the rest are free. JSON inspection and rule authoring are pure file I/O.
- **Failure mode.** If the agent finishes without satisfying the hard constraint, the output JSON should still be written with `match_rate_on_sampled < 1.0` and the rationale explaining why. Downstream pipelines can detect and either fall back to the algorithmic gen or re-run with a larger budget.
- **Comparison runs.** Persist outputs to `rules/.../agent/opus47/agentic/raw/` (Task 1, random sample) and `rules/.../agent/opus47/agentic_fps/raw/` (Task 2, FPS sample) — both separate from the pre-existing `rules/.../agent/opus47/raw/` (the older `rule_gen_agent_claude` pipeline) so they can be compared per question on both `D_s` and `D_u`.

---

## 11. Open questions

- **How much does verifier-grounded generation beat one-shot generation?** Comparable: this pipeline vs `rule_gen_agent_claude` (same JSON input, but no shared verification tools — `verify_accuracy --d-star-mode all_labeled` is the new ingredient). Recommend a side-by-side ablation on the 10 sampled questions.
- **Does FPS-selected D_s generalise better than the random sample?** The Task 1 / Task 2 split is exactly this ablation: same pipeline, different 10-doc training set; compare uAcc on the matching unsampled split.
- **Should `compute_coverage` compute on-demand for fresh rules?** Currently it returns 0.0 when no cache exists, and the agent derives `cov(r)` from a single-rule `verify_accuracy` call (which counts against the budget). A `--compute-on-miss` flag could decouple `cov(r)` from the verify budget at the cost of more gpt54 calls. Tune empirically.

---

## 12. Implementation status

Implemented. Files:

- `tools/list_docs.py`, `tools/read_doc_json.py`, `tools/write_rule.py` — the new tools.
- `tools/verify_accuracy.py` — gained `--d-star-mode {file, all_labeled}` so generation can use `D*_s = D_s` (all sampled docs).
- `agent/run_agent_gen.py` — the per-question driver. CLI: `--sample-set {random, fps}`, `--budget`, `--model opus47`, `--slug`, `--dry-run`.
- `agent/task_prompt_gen.md` — the task prompt template.
- `test/run_eval_merge_agentic.py` — downstream evaluator, parameterised on `--sample-set {random, fps}` and `--split {sampled, unsampled}`.

The deltas vs `agent/run_agent_select.py`:

1. Tools used: **same five** constraint-verification tools (`compute_cost`, `compute_coverage`, `verify_accuracy`, `list_rules`, `inspect_rule`), **plus three new ones** (`list_docs`, `read_doc_json`, `write_rule`). `verify_accuracy` is invoked with `--d-star-mode all_labeled` instead of reading an eval_merge file.
2. Task prompt template changes accordingly: same hard / soft constraints, but with a JSON-inspection + `write_rule` block replacing the pool-browsing block. Output schema mirrors selection-agentic's so downstream summary scripts can read both.
3. Output directory tree: Task 1 (random sample) lands in `rules/financebench/lsf/single_cluster/agent/opus47/agentic/raw/<slug>_10_agentic/`; Task 2 (FPS) in `.../agentic_fps/raw/<slug>_10_agentic_fps/`. Both leave the pre-existing `agent/opus47/raw/` (older `rule_gen_agent_claude`) untouched.

Everything else (subprocess call, token capture, trace JSONL, AGENTIC_*_DONE summary line) is reused verbatim from the selection driver.

---

## 13. Summary

Agentic rule generation replicates `rule_selection_agentic.md` exactly — same Claude Opus 4.7 outer loop, same per-question session, same captured trace, **same hard / soft constraints** (`A(R,d)=1` for `d∈D*_s`; minimise `Σ avg_cost_ratio`, maximise `min cov`, keep `|R|` small), and **the same five verification tools** (`compute_cost`, `compute_coverage`, `verify_accuracy`, `list_rules`, `inspect_rule`). The only differences are the **inputs** (the reconstructed JSON corpus instead of a pre-built rule pool), the **added tools** (`list_docs` + `read_doc_json` + `write_rule`), and the **outputs** (newly-authored Python rule files plus a session-summary JSON).

The expected niche is **datasets where no high-quality rule pool exists yet**, and the agent must invent one — using the same direct accuracy signal (`verify_accuracy`) that drove the selection pipeline.
