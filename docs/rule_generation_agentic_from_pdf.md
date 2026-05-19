# Agentic Rule Generation from PDFs

This document specifies a Claude Code–driven agentic approach to **rule generation from scratch**. The agent receives **only PDFs, the question, and ground-truth labels** — no pre-existing rules of any kind. Its task is to **invent the rule set** by inspecting PDFs and writing new Python rule functions, iterating against the constraints until the merge accuracy target is met.

It is the generation-side counterpart of `docs/rule_selection_agentic.md`. The two specs differ in input and output:

| | `rule_selection_agentic.md` | **this spec** |
|---|------------------------------|----------------|
| Input | Pre-generated rule pool + reconstructed JSON | **Only PDFs + question + labels (no rules at all)** |
| Agent's job | Pick a subset from the existing pool | **Write new rule code from scratch** |
| Output | A list of rule names already in the pool | **New `.py` files containing rule functions** |

The driver mirrors `agent/run_agent_select.py`'s pattern: spawn one Claude Opus 4.7 session per question, expose a fixed set of tools the agent invokes via Bash, capture the result + trace per session. The differences are the **inputs**, the **tools** (PDF inspection + rule authoring instead of pool browsing), and the **output** (rule `.py` files instead of a selection JSON).

---

## 1. Goal

Use Claude Code with Opus 4.7 as an agent that **generates a minimal set of Python span-retrieval rules from scratch** for each question. The agent's session starts with **no rules of any kind** — its only inputs are the PDFs, the question text, and the ground-truth labels. It must inspect the PDFs, identify where the answer lives, and write Python rule functions that retrieve that content. The hard merge-accuracy constraint and the soft cost / rule-count targets define when it can stop.

The output rules follow the same signature and downstream-compatibility contract as the existing LLM-coarse and agent-raw pipelines: a Python file per rule, each exporting a `def rule_<name>(doc: dict) -> list[dict]` that retrieves spans matching some pattern. This means the generated rules plug into the existing `rule_apply_merge` / `eval_judge` / Pareto-selection / agentic-selection pipelines unchanged.

---

## 1.5. Inputs and outputs (explicit)

### What the agent has at session start

| Item | Source |
|------|--------|
| The question text | `data/financebench/sample_queries.txt` (one line, passed to the agent in its prompt) |
| The sampled PDFs (10 files) | `data/financebench/pdf/sampled/*.pdf` |
| Ground-truth labels for the sampled docs | `data/financebench/sample_doc_labels.json` (read by `verify_accuracy` and `compute_coverage`, not directly by the agent) |
| The hard constraint and soft targets | Spelled out in the task prompt |
| The set of tools | PDF inspection + rule authoring + rule testing — see §4 |

### What the agent does NOT have at session start

| Item | Why excluded |
|------|--------------|
| Any rule files | The agent generates them. The output directory starts empty. |
| Any pre-built rule pool | This is the key distinction from `rule_selection_agentic.md`. |
| Reconstructed JSON spans (unless it explicitly requests one via `reconstruct_pdf`) | Reconstruction is a tool, not a free input. Cached after first use. |
| Coverage / cost statistics | No `cov(r)` to look up — these only exist after a rule is written and tested. |

### What the agent produces

| Output | Path |
|--------|------|
| Generated rule files (one per rule) | `rules/.../agent/opus47_pdf/raw/<slug>_10_agent_pdf/rule_<name>.py` |
| Session summary | `results/.../selected_rules_gen/<slug>.json` (rule list, metrics, rationale) |
| Per-tool-call trace | `results/.../agent_trace/<slug>.jsonl` |

The output rule files are the only artifact downstream pipelines need. They drop straight into existing infrastructure (`rule_apply_merge`, Pareto selection, agentic selection, etc.).

---

## 2. Problem setting

The corpus is now a **directory of PDFs**, not a directory of reconstructed JSON files. Concretely:

- **Sampled docs** `D_s`: 10 PDFs (calibration set, with ground-truth answers in a labels file).
- **Unsampled docs** `D_u`: 50 PDFs (held-out generalization set, also with ground-truth answers).
- **Reconstructed JSON**: produced on-demand by a reconstruction tool when the agent or a rule needs to test against the structured representation. The agent doesn't see the JSON unless it explicitly requests it.

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

### Additional tools needed by generation-from-PDF (not in selection-agentic)

The five tools above are shared. Generation adds these because the agent needs to inspect PDFs (rather than browse a pre-built rule pool) and must persist new rule files.

### 4.6 `list_pdfs(directory)` — free

Returns the list of PDF paths in the sampled directory, with per-doc page count and size. Used to know what documents are available.

```bash
python tools/list_pdfs.py --dir data/financebench/pdf/sampled
```

### 4.7 `read_pdf_pages(pdf, pages)` — free

Extracts text from a page range (via PyMuPDF / pdfplumber), preserving rough layout.

```bash
python tools/read_pdf_pages.py --pdf <path> --pages 1-3
```

### 4.8 `read_pdf_vision(pdf, page, query)` — paid (multimodal LLM)

Renders a page as an image and asks gpt-4o-vision (or similar) about layout, fonts, tables. Use sparingly — text extraction is usually sufficient.

```bash
python tools/read_pdf_vision.py --pdf <path> --page 1 --query "What sections appear on this page?"
```

Wraps `src/tools/process_page_image.py`.

### 4.9 `reconstruct_pdf(pdf)` — free, deterministic, cached

Runs the PDF → JSON reconstruction pipeline to produce span-level structured data. Required for `compute_cost` / `compute_coverage` / `verify_accuracy` to test newly-written rules. The first call on a PDF reconstructs; subsequent calls read the cache.

```bash
python tools/reconstruct_pdf.py --pdf <path>
```

### 4.10 `write_rule(name, code)` — free

Persists a new rule to `rules/<question_slug>/<name>.py`. Validates the `def rule_<name>(doc: dict) -> list[dict]` signature and runs a smoke import to catch syntax errors.

```bash
python tools/write_rule.py --question-slug <slug> --name <rule_name> --code-file /tmp/proposed_rule.py
```

After this call, the rule is visible to `list_rules`, `inspect_rule`, `compute_cost`, `compute_coverage`, and `verify_accuracy` — all five constraint-verification tools operate on it identically to how they'd operate on a pre-existing pool rule.

---

### Tool tiers (cost discipline)

| Tier | Tools |
|------|-------|
| Free, no LLM | `compute_cost`, `compute_coverage` (when cached), `list_rules`, `inspect_rule`, `list_pdfs`, `read_pdf_pages`, `reconstruct_pdf`, `write_rule` |
| Paid, cheap | `read_pdf_vision` (one page at a time), `compute_coverage` (when not cached — runs LLM per missing rule) |
| Paid, expensive | `verify_accuracy` (~20 gpt54 calls per invocation; budget-capped, default 30/Q) |

`verify_accuracy` is the **only** tool capped by the per-question budget, matching selection-agentic's policy.

---

## 5. Agent loop

Mirrors `docs/rule_selection_agentic.md` §5 — same constraint/objective formulation, same verification tools. The only differences are (a) the rule set starts empty and grows via `write_rule`, (b) the inspection step uses PDF tools instead of `list_rules`/`inspect_rule` on a pre-built pool. After a rule is written, the constraint-checking tools (`compute_cost`, `compute_coverage`, `verify_accuracy`) work identically to the selection case.

High-level loop per question:

```
1. list_pdfs(sampled_dir)                            # see what's available
2. read_pdf_pages(pdf_a, [1, 2])                     # inspect representative PDFs
   (optional) read_pdf_vision(pdf_a, page=N, query)  # vision if text is insufficient
3. reconstruct_pdf(pdf_a)                            # PDF → JSON (cached); needed
                                                     # so constraint tools can run

4. Propose an initial rule based on observed patterns
5. write_rule(<name>, <code>)                        # rule now persisted

6. compute_cost(rules=[<name>])                      # free; per-rule cost
   compute_coverage(rules=[<name>])                  # free if cached; cov(r)

7. While hard constraint not met OR soft targets unsatisfied:
       verify_accuracy(rules=R)                      # paid; the hard-constraint check
       inspect missed docs (read_pdf_pages on docs
           where match_rate < 1)
       Decide: write a new rule, modify an existing rule,
           or drop a rule
       (write_rule / inspect_rule / overwrite via Write)
       Re-check soft targets via compute_cost / compute_coverage

8. Report final R with cost / coverage / accuracy summary
```

The agent's freedom — same as the selection-agentic version, with two additions:

- **Visual inspection**: `read_pdf_vision` for a confusing page → describes layout, fonts, tables.
- **Cross-doc reasoning**: read 2–3 PDFs before writing a rule, so the rule generalises rather than overfits one layout.
- **Adaptive specificity**: start broad, specialise only for docs not yet covered.
- **Explanation**: each rule's docstring is written by the agent, captured in `inspect_rule`'s output.

Cost-controlled inner / outer loop, matching selection-agentic:

- **Cheap inner loop**: `compute_cost`, `compute_coverage` (with cache), `list_rules`, `read_pdf_pages`, `reconstruct_pdf`, `write_rule`. The agent iterates here freely.
- **Expensive outer check**: `verify_accuracy` (gpt54 QA + judge over the merged retrieval). Used only when the agent thinks the rule set is complete. Budget-capped at 30 calls/Q (matching selection §4.3).

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
  compute_coverage.py      # cov(r) per rule — free if cached, else paid (cheap)
  verify_accuracy.py       # the hard-constraint verifier — paid (gpt54 QA + judge)
  list_rules.py            # list rules currently in the question's rule folder
  inspect_rule.py          # read one rule's source

  # added by generation-from-PDF
  list_pdfs.py             # scan directory, gather metadata
  read_pdf_pages.py        # extract text from a page range
  read_pdf_vision.py       # wraps src/tools/process_page_image.py
  reconstruct_pdf.py       # wraps the PDF→JSON reconstruction pipeline (cached)
  write_rule.py            # validate signature + persist a new rule .py file

agent/
  run_agent_gen_from_pdf.py    # outer driver: spawns one Claude session per question
  task_prompt_gen_from_pdf.md  # task prompt template

data/financebench/pdf/
  sampled/                # 10 PDFs (alternative entry point — currently we use processing/ JSONs)
  unsampled/              # 50 PDFs

results/financebench_single_cluster/agent/opus47_pdf/raw/
  <question_slug>_10_agent_pdf/<rule_name>.py    # generated rules
  agent_trace/<slug>.jsonl                       # per-step tool-call trace
  selected_rules_gen/<slug>.json                 # session summary (rule list + rationale + tokens)
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

The driver fills in `{question}`, `{question_slug}`, `{pdf_sampled_dir}`, `{output_dir}`, `{budget}` placeholders in `task_prompt_gen_from_pdf.md` and captures stdout for the AGENTIC_GEN_DONE summary line.

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

## 8. Task-prompt template (`agent/task_prompt_gen_from_pdf.md`)

Mirrors `agent/task_prompt.md` (the selection prompt) — same hard / soft constraint wording, same verification tools — with the rule-pool browsing block replaced by a PDF-inspection block and a `write_rule` step.

```
You are working inside the LSF project root. Your task is to GENERATE a small
set of Python span-retrieval rules from scratch for the following question,
by inspecting the sampled PDF documents and writing new rule files.

QUESTION   : {question}
SLUG       : {question_slug}
SAMPLED PDFs: {pdf_sampled_dir}     (10 PDFs)
LABELS     : data/financebench/sample_doc_labels.json
OUTPUT DIR : {output_dir}/{question_slug}_10_agent_pdf/   (write rules here)
TRACE      : {trace_path}

HARD CONSTRAINT (must be satisfied before you finish)
  Merge accuracy of your generated rule set R must equal the merge accuracy of
  perfect retrieval on every sampled doc:
        A(R, d) = 1 for every d in D*_s
  D*_s = the set of sampled docs that have ground-truth labels (= all 10).
  Verifier:
    python tools/verify_accuracy.py --question-slug {question_slug} \
        --question "{question}" --rules <r1> <r2> ...

SOFT TARGETS (negotiate against each other)
  1. Minimise sum(avg_cost_ratio) across R         — tool: compute_cost.py
  2. Maximise min cov(r) across r in R             — tool: compute_coverage.py
  3. Keep |R| small. Prefer one broad rule over three narrow ones.

REASONABLE STOPPING CRITERIA (subjective, optional):
  - min_cov(R) >= 0.4
  - sum_avg_cost_ratio(R) <= 0.5 * cost_of_naive_full_retrieval
  - |R| <= 10
  Stop when accuracy is met AND any two of these three hold, or when budget
  is exhausted.

TOOLS YOU HAVE (invoke via the Bash tool)

  # Constraint-verification tools — SAME AS rule_selection_agentic
  python tools/compute_cost.py     --question-slug {question_slug} --rules <r1> ...
  python tools/compute_coverage.py --question-slug {question_slug} --rules <r1> ...
  python tools/verify_accuracy.py  --question-slug {question_slug} \
                                    --question "{question}" --rules <r1> ...
  python tools/list_rules.py       --question-slug {question_slug}
  python tools/inspect_rule.py     --question-slug {question_slug} --rule <name>

  # PDF inspection tools — specific to this generation pipeline
  python tools/list_pdfs.py        --dir {pdf_sampled_dir}
  python tools/read_pdf_pages.py   --pdf <path> --pages <range>
  python tools/read_pdf_vision.py  --pdf <path> --page <N> --query "..."   (paid)
  python tools/reconstruct_pdf.py  --pdf <path>                            (cached)

  # Rule authoring tool — specific to this generation pipeline
  python tools/write_rule.py       --question-slug {question_slug} \
                                    --name <rule_name> --code-file <path>

BUDGET: at most {budget} verify_accuracy calls per question (default 30).

LOOP
  1. list_pdfs to see what's available.
  2. Read the first few pages of 2-3 representative PDFs with read_pdf_pages.
     Use read_pdf_vision (paid, sparingly) only if text extraction is unclear.
  3. Form a hypothesis about where the answer lives in a typical filing
     (page band, section header, font properties, table row label).
  4. Reconstruct one PDF first so the downstream tools have JSON to operate on:
        reconstruct_pdf.py --pdf <path>
  5. Author your first rule with write_rule (give it a descriptive name and a
     one-line docstring; signature must be `def rule_<name>(doc: dict) -> list[dict]`).
  6. compute_coverage and compute_cost on your new rule to see how it behaves.
  7. If accuracy not yet at target on D*_s, call verify_accuracy. The per-doc
     verdicts tell you which docs are still missed. Read those PDFs, decide
     whether to:
        - write a new rule for the missed layout (write_rule)
        - revise an existing rule (overwrite with write_rule or the Write tool)
        - drop a rule that's purely overhead
  8. Stop when accuracy is at 1.0 on D*_s AND your soft targets feel reasonable,
     or budget exhausted.

COST AND LATENCY TRACKING (mandatory)

Record time.time() at start. Every verify_accuracy call returns a `tokens`
field; accumulate:
    tool_input_tokens   += result["tokens"]["qa_input"]  + result["tokens"]["j_input"]
    tool_output_tokens  += result["tokens"]["qa_output"] + result["tokens"]["j_output"]
    tool_llm_calls      += 2 * (correct + wrong per doc)
    verify_calls        += 1

Just before writing the final summary, latency_seconds = time.time() - t_start.

OUTPUT

When done, your generated rule .py files already live in
  {output_dir}/{question_slug}_10_agent_pdf/

Write the session summary JSON to
  {output_dir}/selected_rules_gen/{question_slug}.json
with this schema:
{
  "question":              "{question}",
  "question_slug":         "{question_slug}",
  "mode":                  "agentic_gen_from_pdf",
  "model":                 "{model}",
  "selected_rules":        ["rule_a", "rule_b", "..."],
  "selected_avg_cost_ratio_sum": <float>,
  "min_cov":               <float>,
  "mean_cov":              <float>,
  "match_rate_on_sampled": <float>,
  "iterations":            <int>,
  "verify_calls":          <int>,
  "tool_llm_calls":        <int>,
  "tool_input_tokens":     <int>,
  "tool_output_tokens":    <int>,
  "latency_seconds":       <float>,
  "rationale":             "<2-4 sentence explanation>"
}

Append per-tool-call trace to {trace_path}, one JSON line per call:
  { "step": N, "tool": "...", "args": "...", "result_summary": "..." }

Then print to stdout:
  AGENTIC_GEN_DONE slug={question_slug} n_rules=N sum_cost=F \
                   min_cov=F match_rate=F tool_calls=N latency_s=F

GUIDELINES
  - Prefer broad rules whose docstring describes a layout-invariant signal
    over hardcoded position. The hardcoded version overfits.
  - Read at least 2-3 different PDFs before writing your first rule.
  - Use read_pdf_vision ONLY when text extraction is insufficient.
  - When verify_accuracy fails, prefer "add a new rule for the missed layout"
    over "broaden an existing rule" — broadening inflates retrieval cost.
  - Do not refuse to finish. If you cannot satisfy the hard constraint within
    budget, return the best R you have, set match_rate_on_sampled to the actual
    value, and explain in the rationale why.
```

---

## 9. Comparison with the existing rule-gen pipelines

| Aspect | `rule_gen_llm_coarse` | `rule_gen_agent_claude` (current) | **`run_agent_gen_from_pdf`** (this) |
|---|---|---|---|
| Document input | Reconstructed JSON spans | Reconstructed JSON spans | **PDFs directly** (reconstruction on-demand, cached) |
| Outer loop | Single prompt, one shot | Claude inside Claude Code, JSON tools | Claude inside Claude Code, **PDF + JSON tools** |
| Visual inspection | None | None (only sees JSON) | **read_pdf_vision available** for tricky layouts |
| Determinism | Deterministic at temp=0 | Non-deterministic | Non-deterministic |
| Output | Rule pool (~100 rules) | Rule pool (~5-20 rules with rationale) | Rule pool (~3-10 rules with PDF-grounded rationale) |
| When to prefer | Mass rule production, cheap | Hard questions where mass rules underperform | When you only have PDFs (no preprocessed JSON), or when visual layout matters |

The new variant is most useful for **new datasets** where the PDF→JSON reconstruction pipeline hasn't been run, or for questions where rule writers (human or LLM) need to **see the page layout** to write a correct rule.

---

## 10. Operational notes

- **Concurrency.** One Claude Code session per question; sessions are independent and parallelizable subject to model rate limits.
- **Reproducibility.** Even though Opus is non-deterministic, the trace at `agent_trace/<slug>.jsonl` records every tool call. Replaying the same tool sequence is deterministic; only the agent's choices are not.
- **Cost control.** Cap `verify_accuracy` calls per session (default 30, matching `rule_selection_agentic.md` §4.3). The expensive tools are `read_pdf_vision`, uncached `compute_coverage` (one LLM call per missing rule eval), and `verify_accuracy` (~20 gpt54 calls per invocation); the rest are free. Persist the PDF→JSON reconstruction cache so the same PDF is never reconstructed twice.
- **Failure mode.** If the agent finishes without satisfying the hard constraint, the output JSON should still be written with `match_rate_on_sampled < 1.0` and the rationale explaining why. Downstream pipelines can detect and either fall back to the algorithmic gen or re-run with a larger budget.
- **Comparison runs.** Persist outputs to `rules/.../agent/opus47_pdf/raw/` separately from `rules/.../agent/opus47/raw/` so the JSON-input and PDF-input variants can be compared per question on both `D_s` and `D_u`.

---

## 11. Open questions

- **How much does PDF inspection (especially `read_pdf_vision`) add over JSON-only inspection?** The marginal value is hardest to quantify without an ablation. Recommend a small evaluation: run both this PDF-input variant and the existing JSON-input agent on the 10 sampled questions; compare `merge_accuracy_on_unsampled` per question.
- **How aggressively should the agent use `read_pdf_vision`?** It's the most expensive tool. The prompt currently says "only when text extraction is insufficient." Tune empirically.
- **Should the reconstruction step be visible to the agent?** Currently the agent sees `reconstruct_pdf` as a tool it can invoke. An alternative is to have the driver pre-reconstruct all PDFs before launching the agent, so the agent only sees `read_pdf_pages` and `read_pdf_vision`. Less flexibility but cleaner abstraction.

---

## 12. Implementation status

Not yet implemented. The two markdown specs (this one and `docs/rule_selection_agentic.md`) follow the same template, so when this is built it should reuse:

- `tools/_paths.py` — for shared path constants
- `agent/run_agent_select.py` — as a structural template for the driver (subprocess call pattern, `--output-format json`, token capture, AGENTIC_*_DONE summary line)
- `src/tools/process_page_image.py` — already implements vision-based PDF QA; thin wrapper goes in `tools/read_pdf_vision.py`
- The existing PDF→JSON reconstruction pipeline (path TBD in repo) — wrapped by `tools/reconstruct_pdf.py`

The deltas vs `agent/run_agent_select.py`:

1. Tools change: **keep** all five constraint-verification tools (`compute_cost`, `compute_coverage`, `verify_accuracy`, `list_rules`, `inspect_rule`) — they operate on whatever rule files exist in the question folder, whether pre-built or just-written. **Add** five PDF/authoring tools (`list_pdfs`, `read_pdf_pages`, `read_pdf_vision`, `reconstruct_pdf`, `write_rule`).
2. Task prompt template changes accordingly: same hard constraint (`A(R,d)=1` on `D*_s`) and same three soft targets as selection-agentic, but with a PDF-inspection + `write_rule` block in place of the pool-browsing block. Output schema mirrors selection-agentic's so downstream summary scripts can read both.
3. Output directory tree changes: rules land in `rules/.../agent/opus47_pdf/raw/<slug>_10_agent_pdf/`, summary lands in `results/.../selected_rules_gen/<slug>.json`.

Everything else (driver structure, subprocess call, token capture, trace JSONL, AGENTIC_*_DONE summary line) is reused verbatim.

---

## 13. Summary

Agentic rule generation from PDFs replicates `rule_selection_agentic.md` exactly — same Claude Opus 4.7 outer loop, same per-question session, same captured trace, **same hard / soft constraints** (`A(R,d)=1` for `d∈D*_s`; minimise `Σ avg_cost_ratio`, maximise `min cov`, keep `|R|` small), and **the same five verification tools** (`compute_cost`, `compute_coverage`, `verify_accuracy`, `list_rules`, `inspect_rule`). The only differences are the **inputs** (PDFs instead of a pre-built rule pool, with on-demand reconstruction to JSON for testing), the **added tools** (PDF inspection + `write_rule`), and the **outputs** (newly-authored Python rule files in addition to the session-summary JSON).

The expected niche is **new datasets where no reconstructed JSON yet exists**, or hard questions where **visual layout inspection** (`read_pdf_vision`) helps the agent generalize across template families that JSON-only inspection misses.
