# Agentic Rule Generation from PDFs

This document specifies a Claude Code–driven agentic approach to **rule generation** where the input is **PDF files directly**, with no pre-existing reconstructed-JSON intermediate required. It is the generation-side counterpart of `docs/rule_selection_agentic.md` (which covers selection over an already-generated rule pool).

The driver mirrors `agent/run_agent_select.py`'s pattern: spawn one Claude Opus 4.7 session per question, expose a fixed set of tools the agent invokes via Bash, capture the result + trace per session. The only differences from rule-selection-agentic are the **inputs** (PDFs instead of reconstructed JSON + a pre-built rule pool), the **tools** (PDF inspection tools instead of `list_rules`/`inspect_rule`), and the **output** (rule `.py` files instead of a selected-rules JSON).

---

## 1. Goal

Use Claude Code with Opus 4.7 as an agent that generates a minimal set of Python span-retrieval rules for each question, given **PDFs as the only document input**. The agent decides what rules to write, when to broaden or specialize them, and when to stop, subject to a hard merge-accuracy constraint and soft cost / rule-count targets.

The output rules follow the same signature and downstream-compatibility contract as the existing LLM-coarse and agent-raw pipelines: a Python file per rule, each exporting a `def rule_<name>(doc: dict) -> list[dict]` that retrieves spans matching some pattern. This means the generated rules plug into the existing `rule_apply_merge` / `eval_judge` / Pareto-selection / agentic-selection pipelines unchanged.

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

| Type | Constraint | How the agent checks it |
|---|---|---|
| Hard | Merge accuracy on `D_s` ≥ `target_accuracy` (default 0.95) | Call the `test_union` tool (LLM-as-judge over merged retrieval of all rules in the current set) |
| Soft target | Minimise `Σ_{r∈R} avg_cost_ratio(r)` (retrieved tokens / total tokens) | Call the `test_rule` tool, which reports per-rule cost |
| Soft target | Minimise `|R|` — prefer one broad rule over three narrow ones | Track in the agent's working memory |
| Soft target | Per-rule coverage (fraction of `D_s` the rule fires on correctly) ≥ 0.2 | `test_rule` reports per-rule per-doc retrieval; agent computes coverage |

The hard constraint is the only one the agent cannot finalize without satisfying. Soft targets are negotiated against each other; the agent should aim for a defensible trade-off and explain its choice in the final report.

A rule's cost is `retrieved_tokens / total_doc_tokens` per doc, averaged across `D_s`. Lower = more selective retrieval. Coverage is measured by a fast substring proxy against ground truth; only the final union accuracy is judged by LLM.

---

## 4. Tools the agent needs

The five tools below wrap primitives already in `src/tools/` plus a couple of new ones for the PDF-side workflow. Each is a thin CLI invoked via the `Bash` tool.

### 4.1 `list_pdfs(directory)` — free, no LLM

Returns the list of PDF paths in the sampled or unsampled directory, plus per-doc metadata (page count, file size). The agent uses this to know what documents are available.

```bash
python tools/list_pdfs.py --dir data/financebench/pdf/sampled
```

Output (JSON):
```json
{
  "directory": "...",
  "pdfs": [
    {"name": "AMCOR_2019_10K.pdf", "pages": 138, "size_mb": 2.4},
    ...
  ]
}
```

### 4.2 `read_pdf_pages(pdf, pages=[1,2,3])` — free, no LLM

Renders the requested page range as **text** (extracted via PyMuPDF or pdfplumber), preserving rough layout (line breaks, table structure). For the agent to inspect what a page contains.

```bash
python tools/read_pdf_pages.py --pdf <path> --pages 1-3
```

Output: concatenated page text with `---PAGE 1---` separators.

### 4.3 `read_pdf_vision(pdf, page=N)` — paid (uses Opus / gpt-4o vision)

Renders a PDF page as an image and asks a multimodal model to describe layout, fonts, tables. Use sparingly; useful when text extraction misses visual structure (e.g. complex tables, marginalia, charts).

```bash
python tools/read_pdf_vision.py --pdf <path> --page 1 --query "What sections appear on this page?"
```

Wraps the existing `src/tools/process_page_image.py`.

### 4.4 `reconstruct_pdf(pdf)` — free (deterministic), cached

Runs the standard PDF→JSON reconstruction pipeline to produce a span-level structured JSON (the same format `data/financebench/processing/*.json` uses). Required for testing rules — rules operate on JSON spans, not on PDFs directly.

```bash
python tools/reconstruct_pdf.py --pdf <path> --out /tmp/recon/<name>.json
```

This step is deterministic and cacheable. The first agent that touches a given PDF reconstructs once; subsequent tool calls read the cache.

### 4.5 `test_rule(rule_path, sampled_pdfs)` — paid (substring proxy, no LLM unless `--judge` flag set)

Loads a single rule `.py` file, applies it across the sampled docs (via the reconstructed JSONs), and reports per-doc:

- `retrieved_text` (truncated)
- `retrieved_tokens / total_tokens` (cost ratio)
- `ground_truth_substring_present` (fast proxy for correctness)
- Optionally: `llm_judge_result` if `--judge` is passed (the only LLM call here)

```bash
python tools/test_rule.py --rule rules/<question_slug>/<rule_name>.py \
                          --sampled-dir data/financebench/pdf/sampled \
                          [--judge]
```

The substring proxy is the cheap inner-loop signal; the agent reserves `--judge` for final validation.

### 4.6 `test_union(rule_paths, sampled_pdfs)` — paid (LLM judge)

Like `test_rule` but for the **union** of multiple rules: applies all listed rules to each doc, concatenates the retrieved spans, sends to gpt54 with the question, judges the answer against ground truth. Returns per-doc verdicts and the overall merge accuracy.

```bash
python tools/test_union.py --rules rule_a.py rule_b.py rule_c.py \
                            --question-slug what_is_...
```

This is the **hard-constraint verifier**: the agent cannot finalize until `test_union` reports `merge_accuracy ≥ target_accuracy`.

### 4.7 `write_rule(name, code)` — free

The agent writes a candidate rule to `rules/<question_slug>/<name>.py`. The tool validates the rule's signature (`def rule_<name>(doc: dict) -> list[dict]`), runs a smoke import to catch syntax errors, and writes the file.

```bash
python tools/write_rule.py --question-slug <slug> --name <rule_name> --code-file /tmp/proposed_rule.py
```

Alternatively, the agent can write the file directly with the standard `Write` tool — `write_rule.py` just adds the signature check.

The first six tools are sufficient for the inner loop. `write_rule` is the persistence step.

---

## 5. Agent loop

The high-level pseudocode per question:

```
1. list_pdfs(sampled_dir)                          # see the corpus
2. read_pdf_pages(pdf_a, [1, 2])                   # inspect a representative doc
3. (optional) read_pdf_vision(pdf_a, page=N)       # if text extraction is unclear
4. propose initial rule based on observed patterns
5. write_rule(<name>, <code>)
6. test_rule(<rule>, sampled_pdfs)                 # substring proxy across all sampled
7. Loop:
       If individual coverage too low → inspect more PDFs, refine the rule
       If individual coverage acceptable but some docs not covered →
           propose an additional rule targeted at the missed docs
           (read those PDFs first to understand their structure)
       Update rule set, call test_union to check the hard constraint
       If hard constraint met AND soft targets settled → exit
8. Report final rule set with rationale.
```

The agent's freedom (vs a hand-coded greedy):

- **Visual inspection**: it can call `read_pdf_vision` on a confusing page and use the model's description to decide what features matter.
- **Cross-doc reasoning**: after seeing two PDFs, it can write a rule that generalizes (e.g., "first bold span on page 1 whose font is in the Helvetica family") rather than overfitting to one layout.
- **Adaptive specificity**: it can start with broad rules and only specialize when broad ones miss specific docs.
- **Explanation**: each rule's docstring is written by the agent, capturing intent for downstream review.

The cost-controlled loop: `test_rule` with substring proxy is the cheap inner loop. `test_union` with LLM judge is the expensive outer check, used only when the agent thinks the rule set is complete.

---

## 6. Termination

The agent stops when either:

- **Success**: `test_union` reports `merge_accuracy ≥ target_accuracy` (default 0.95), *and* the soft targets are at a reasonable trade-off. Reasonable can be operationalized as: (a) `|R| ≤ 10`, (b) `sum_avg_cost_ratio(R) ≤ 0.05` (5% of doc tokens), (c) per-rule coverage `≥ 0.2`. If all three soft conditions hold and accuracy is met, return.
- **Stuck**: 5 consecutive iterations with no improvement in merge accuracy. Return the best `R` seen so far with a "could not improve further" note.
- **Budget exhausted**: total `test_union` calls exceed a configured budget (default 20 per question, so ~200 LLM-judge invocations across the 10 sampled docs × 20 union evals).

The agent must produce a final report including the rule list, all three metric values, and a one-paragraph rationale.

---

## 7. Implementation outline

### 7.1 Files to add

```
tools/
  list_pdfs.py             # 4.1 — scan directory, gather metadata
  read_pdf_pages.py        # 4.2 — extract text from a page range
  read_pdf_vision.py       # 4.3 — wraps src/tools/process_page_image.py
  reconstruct_pdf.py       # 4.4 — wraps the PDF→JSON reconstruction pipeline
  test_rule.py             # 4.5 — applies one rule, substring proxy + optional LLM judge
  test_union.py            # 4.6 — applies a rule set, full LLM judge
  write_rule.py            # 4.7 — validates signature + writes the .py file

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

`selected_rules_gen/<slug>.json`:

```json
{
  "question":               "...",
  "question_slug":          "...",
  "mode":                   "agentic_gen_from_pdf",
  "model":                  "claude-opus-4-7",
  "rule_files":             ["rule_<name>.py", "rule_<name>.py", ...],
  "num_rules":              3,
  "merge_accuracy":         0.95,
  "avg_cost_ratio":         0.032,
  "min_per_rule_coverage":  0.6,
  "iterations":             7,
  "test_union_calls":       4,
  "tool_llm_calls":         X,
  "tool_input_tokens":      X,
  "tool_output_tokens":     X,
  "latency_seconds":        X,
  "rationale":              "After inspecting AMCOR p1 + p2 (rendered with read_pdf_vision), wrote a broad rule for cover-page bold spans. test_union failed on EBAY; inspected EBAY p1 — added a sibling rule for the alternative right-column layout. Final 2 rules cover all sampled docs at avg cost 0.032."
}
```

The companion `agent_trace/<slug>.jsonl` records each tool call and its result.

---

## 8. Task-prompt template (`agent/task_prompt_gen_from_pdf.md`)

```
You are working inside the LSF project root. Your task is to generate a minimal
set of Python span-retrieval rules for the following question, by inspecting
the sampled PDF documents directly.

QUESTION   : {question}
SLUG       : {question_slug}
SAMPLED PDFs: {pdf_sampled_dir}    (10 PDFs)
LABELS     : data/financebench/sample_doc_labels.json  (ground truth per doc)
OUTPUT DIR : {output_dir}/{question_slug}_10_agent_pdf/   (write rules here)
TRACE      : {trace_path}

HARD CONSTRAINT (must be satisfied before you finish)
  Merge accuracy of the union of your rules, judged by gpt54 against ground
  truth, must reach {target_accuracy} on the sampled docs.
  Verifier: python tools/test_union.py --rules <r1>.py <r2>.py ... \
                --question-slug {question_slug}

SOFT TARGETS (negotiate against each other)
  1. Minimise sum(avg_cost_ratio) across your rules.
  2. Keep |R| small. Prefer one broad rule over three narrow ones.
  3. Each rule should fire correctly on ≥ 20% of sampled docs (per-rule cov).

REASONABLE STOPPING CRITERIA (subjective, optional):
  - merge_accuracy ≥ {target_accuracy}
  - sum_avg_cost_ratio ≤ 0.05
  - |R| ≤ 10
  Stop when accuracy is met AND any two of (1), (2), (3) hold above.

TOOLS YOU HAVE (invoke via the Bash tool)

  # Free / cheap:
  python tools/list_pdfs.py        --dir {pdf_sampled_dir}
  python tools/read_pdf_pages.py   --pdf <path> --pages <range>
  python tools/reconstruct_pdf.py  --pdf <path>           # PDF → JSON; cached
  python tools/test_rule.py        --rule <path>          # substring proxy
  python tools/write_rule.py       --name <name> --code-file <path>

  # Paid (use sparingly):
  python tools/read_pdf_vision.py  --pdf <path> --page <N> --query "..."
  python tools/test_rule.py        --rule <path> --judge   # adds gpt54 judge
  python tools/test_union.py       --rules <r1>.py <r2>.py ...

BUDGET: at most {budget} test_union calls per question (default 20).

LOOP
  1. list_pdfs to see what's available.
  2. Pick a representative doc; read_pdf_pages on its first few pages to
     understand structure. If the text extraction looks garbled or you need to
     see layout, use read_pdf_vision (paid).
  3. Form a hypothesis about where the answer to {question} lives in a typical
     filing (page band, section header, font properties).
  4. write_rule with a first-draft rule expressing that hypothesis.
  5. test_rule against the sampled set (substring proxy is free; use it).
     - If coverage is low (< 0.5), inspect a doc the rule missed; revise.
     - If coverage is decent (≥ 0.5) but some docs aren't covered, add an
       additional rule targeting those.
  6. When you think the rule set is complete, call test_union to validate.
     If merge_accuracy < target, the per-doc verdicts tell you which docs
     are still wrong — inspect those, refine, retry.
  7. Stop when accuracy is met AND soft targets are reasonable, or budget
     exhausted.

COST AND LATENCY TRACKING (mandatory)

Record time.time() at start. Accumulate gpt54 token counts from every
test_rule --judge and test_union call (each tool prints a `tokens` field in
its JSON output). Just before writing the final summary, compute
latency_seconds = t_end - t_start.

OUTPUT

When done, write rule files to {output_dir}/{question_slug}_10_agent_pdf/
(use write_rule.py, one file per rule). Then write the session summary JSON
to {output_dir}/selected_rules_gen/{question_slug}.json with this schema:
  { "rule_files": [...], "num_rules": N, "merge_accuracy": F,
    "avg_cost_ratio": F, "min_per_rule_coverage": F, "iterations": N,
    "test_union_calls": N, "tool_llm_calls": N, "tool_input_tokens": N,
    "tool_output_tokens": N, "latency_seconds": F, "rationale": "..." }

Append per-tool-call trace to {trace_path} (one JSON line per call):
  { "step": N, "tool": "...", "args": "...", "result_summary": "..." }

Then print a single AGENTIC_GEN_DONE line to stdout:
  AGENTIC_GEN_DONE slug={question_slug} num_rules=N merge_acc=F \
                   avg_cost=F latency_s=F

GUIDELINES
  - PREFER rules whose docstring describes a layout-invariant signal (e.g.
    "first H1 span on page 1 with font size > 14") OVER hardcoded position
    (e.g. "span at index 17 on page 1"). The hardcoded version overfits.
  - Read at least 2-3 different PDFs before writing your first rule. Different
    filers use different templates; rules built from one doc rarely transfer.
  - Use read_pdf_vision ONLY when text extraction is insufficient. It costs
    LLM tokens; the text tools are free.
  - When test_union fails, prefer "add a new rule for the missed layout" over
    "broaden an existing rule" — broadening tends to inflate retrieval cost.
  - Do not refuse to finish. If you cannot meet target accuracy within budget,
    return the best rule set, set merge_accuracy to the actual value, and
    explain in the rationale why.
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
- **Cost control.** Cap `test_union` calls per session (default 20). The expensive tools are `read_pdf_vision`, `test_rule --judge`, and `test_union`; the rest are free. Persist the PDF→JSON reconstruction cache so the same PDF is never reconstructed twice.
- **Failure mode.** If the agent finishes without meeting target accuracy, the output JSON should still be written with `merge_accuracy < target` and the rationale explaining why. Downstream pipelines can detect and either fall back to the algorithmic gen or re-run with a larger budget.
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

1. Tools change: drop `list_rules`/`inspect_rule`/`compute_cost`/`compute_coverage`/`verify_accuracy`; add `list_pdfs`/`read_pdf_pages`/`read_pdf_vision`/`reconstruct_pdf`/`test_rule`/`test_union`/`write_rule`.
2. Task prompt template changes accordingly (different hard constraint, different toolset, different output schema).
3. Output directory tree changes: rules land in `rules/.../agent/opus47_pdf/raw/<slug>_10_agent_pdf/`, summary lands in `results/.../selected_rules_gen/<slug>.json`.

Everything else (driver structure, subprocess call, token capture, trace JSONL, AGENTIC_*_DONE summary line) is reused verbatim.

---

## 13. Summary

Agentic rule generation from PDFs replicates the structure of agentic rule selection — Claude Opus 4.7 as the outer-loop optimizer, tool-mediated interaction via Bash, per-question session, captured trace — but the **inputs are PDFs** (with on-demand reconstruction to JSON for testing) and the **outputs are Python rule files** (not a selected-rules JSON). The toolset shifts from rule-pool-inspection tools to PDF-inspection tools, and the hard constraint shifts from "match `D*` coverage" to "merge_accuracy ≥ target on `D_s`."

The expected niche is **new datasets where no reconstructed JSON yet exists**, or hard questions where **visual layout inspection** (`read_pdf_vision`) helps the agent generalize across template families that JSON-only inspection misses.
