You are working inside the LSF project root. Your task is to GENERATE a small
set of Python span-retrieval rules FROM SCRATCH for the following question,
by inspecting the reconstructed JSON representations of the sampled
documents and writing new rule files. You are NOT selecting from a pool — the
output folder starts empty and you write every rule in it.

QUESTION   : {question}
SLUG       : {question_slug}
LABELS     : {labels_file}
PROCESSING : {processing_dir}   (reconstructed JSON lives here, one per doc)
RULES DIR  : {rules_dir}        (write rules to <rules_dir>/{question_slug}/)
OUTPUT     : {output_path}      (your final session-summary JSON)
TRACE      : {trace_path}
COST CACHE : {cost_cache_dir}
COV CACHE  : {eval_individual_dir}

HARD CONSTRAINT (must be satisfied before you finish)
  For every sampled doc d, the merge accuracy of your rule set R must be 1.
  Concretely: verify_accuracy.py with --d-star-mode all_labeled must return
  `missed_in_D_star: []` (equivalently `match_rate = 1.0`).

SOFT TARGETS (negotiate against each other — may be violated to satisfy the hard constraint)
  1. Minimise sum(avg_cost_ratio) across R         (tool: compute_cost.py)
  2. Maximise min cov(r) across r in R             (use verify_accuracy --rules <r>
                                                    on a single rule; that match_rate
                                                    IS cov(r). compute_coverage.py
                                                    returns 0.0 for newly-written rules.)
  3. Keep |R| small. Prefer one broad rule over three narrow ones.

REASONABLE STOPPING CRITERIA (subjective, optional):
  - min_cov(R) >= 0.4
  - sum_avg_cost_ratio(R) <= 0.05
  - |R| <= 10
  Stop when the hard constraint holds AND any two of these three are satisfied,
  or when budget is exhausted.

TOOLS YOU HAVE (invoke via the Bash tool, one per call)

  # ── JSON inspection (free) ────────────────────────────────────────────────
  python tools/list_docs.py --labels-file {labels_file}
  python tools/read_doc_json.py --doc <stem> --page <N>
  python tools/read_doc_json.py --doc <stem> --pages 1-3 --filter "<substr>" --max-spans 80

  # ── Rule authoring (free) ─────────────────────────────────────────────────
  python tools/write_rule.py --question-slug {question_slug} --name rule_<name> \
      --code-file /tmp/<name>.py --rules-dir {rules_dir}
  # (use --overwrite to replace an existing rule)

  # ── Pool inspection (free) ────────────────────────────────────────────────
  python tools/list_rules.py    --question-slug {question_slug} --rules-dir {rules_dir}
  python tools/inspect_rule.py  --question-slug {question_slug} --rule <name>      \
                                --rules-dir {rules_dir}

  # ── Soft-target metrics ───────────────────────────────────────────────────
  # compute_cost (free, no LLM)
  python tools/compute_cost.py --question-slug {question_slug} \
      --rules <r1> <r2> ...        \
      --rules-dir   {rules_dir}    \
      --labels-file {labels_file}  \
      --cache-dir   {cost_cache_dir}

  # ── Hard-constraint verifier (paid, ~20 gpt54 calls per invocation) ──────
  python tools/verify_accuracy.py --question-slug {question_slug} \
      --question "{question}" --rules <r1> <r2> ... \
      --rules-dir     {rules_dir}     \
      --labels-file   {labels_file}   \
      --output-dir    {selector_run_dir} \
      --d-star-mode   all_labeled

BUDGET: at most {budget} verify_accuracy calls. Use them sparingly. To check
the coverage of a single new rule, you can call verify_accuracy with
--rules <single_rule_name>; the resulting `match_rate` equals cov(r). That
counts against your verify budget too, so prefer to verify the full set first
and only probe single-rule coverage when triaging a failing union.

LOOP (suggested; deviate if you have a better idea)

  1. list_docs → see the 10 reconstructed JSONs and their span/page counts.
  2. read_doc_json on 2-3 representative docs to understand the structure
     (cover page, section headers, tables). Use --page or --filter to focus.
  3. Form a hypothesis about where the answer to "{question}" lives in a
     typical filing. Patterns that work well:
       - cover-page H1 spans with bold + large size
       - structure.path_text matches a known section header
       - cell adjacent to a label string (e.g. text == "Telephone Number")
       - first numeric span on the page following a given header
  4. Author your first rule with write_rule. Give it a descriptive name
     (e.g. rule_cover_page_company_name) and a one-line docstring describing
     the layout signal. Signature MUST be exactly:

       def rule_<name>(doc: dict) -> list[dict]:
           '''<one-line description>'''
           return [span for span in doc.get("texts", []) if ...]

     A returned span MUST be an element from doc["texts"] (do not synthesise).
  5. compute_cost on your rule to see how much it retrieves.
  6. verify_accuracy --rules <r1> on the singleton — this gives both the
     hard-constraint signal so far and effectively cov(r1).
  7. If the union still misses some docs:
       - read_doc_json on a missed doc to find the alternative layout.
       - Either add a sibling rule (preferred) or revise the existing rule
         (write_rule --overwrite).
     Re-verify until match_rate = 1.0 on all 10 sampled docs.
  8. Once the hard constraint holds, consider drop-tests: compute_cost
     without each rule and re-verify; drop any rule whose absence keeps
     match_rate = 1.0.
  9. Stop when hard constraint holds AND soft targets feel reasonable, or
     budget exhausted.

COST AND LATENCY TRACKING (mandatory)

Record `time.time()` immediately when you start. Each verify_accuracy result
contains a `tokens` field. Accumulate across calls:

    tool_input_tokens   += result["tokens"]["qa_input"]  + result["tokens"]["j_input"]
    tool_output_tokens  += result["tokens"]["qa_output"] + result["tokens"]["j_output"]
    tool_llm_calls      += 2 * len([d for d in result["per_doc"] if "error" not in d])
    verify_calls        += 1

Just before writing the final JSON, record `time.time()` again as t_end and
compute latency_seconds = t_end - t_start.

OUTPUT (the final step before exiting)

Your generated rule .py files already live in <rules_dir>/{question_slug}/.
Write a session summary JSON to {output_path} with this exact schema:

{{
  "question":                     "{question}",
  "question_slug":                "{question_slug}",
  "mode":                         "agentic_gen",
  "model":                        "{model}",
  "selected_rules":               ["rule_a", "rule_b", "..."],
  "selected_avg_cost_ratio_sum":  <float>,
  "min_cov":                      <float>,
  "mean_cov":                     <float>,
  "match_rate_on_sampled":        <float>,
  "iterations":                   <int>,
  "verify_calls":                 <int>,
  "tool_llm_calls":               <int>,
  "tool_input_tokens":            <int>,
  "tool_output_tokens":           <int>,
  "latency_seconds":              <float>,
  "rationale":                    "<2-4 sentence explanation of decisions made>"
}}

Then append a per-step JSONL trace to {trace_path} (one object per tool call):
  {{ "step": <int>, "tool": "<name>", "args": "...", "result_summary": "..." }}

Finally, print to stdout a single line:
  AGENTIC_GEN_DONE slug={question_slug} n_rules=<N> sum_cost=<F> \
                   min_cov=<F> match_rate=<F> tool_calls=<N> latency_s=<F>

GUIDELINES
  - Prefer rules whose docstring describes a layout-invariant signal
    (e.g. "first bold H1 span on page 1") over hardcoded positions
    (e.g. "span at index 17"). Hardcoded positions overfit.
  - Read at least 2 different docs before writing your first rule. Filers use
    different templates; rules built from one doc rarely transfer.
  - When verify_accuracy fails, prefer "add a new sibling rule for the missed
    layout" over "broaden an existing rule" — broadening inflates retrieval cost.
  - Rules MUST be self-contained Python: import only from the standard library
    (re, json, etc.) — no third-party packages and no project imports.
  - Do not refuse to finish. If you cannot satisfy the hard constraint within
    budget, return the best R you have, set match_rate_on_sampled to the actual
    value, and explain in the rationale why.

You can call tools in parallel where independent (e.g. inspecting two docs).
You may write small helper Python via Write if needed, but the generated rules
themselves must conform to the signature above.

Work carefully but do not over-think. The 10 sampled docs are uniform enough
that 2-4 rules typically suffice.
