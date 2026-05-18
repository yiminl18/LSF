You are working inside the LSF project root. Your task is to select a small
subset of rules from a pre-generated rule pool that preserves the merge
accuracy of the full pool on the sampled documents, while keeping total cost
low and per-rule coverage high.

QUESTION   : {question}
SLUG       : {question_slug}
RULE POOL  : rules/financebench_single_cluster/llm/gpt54/one_shot/{question_slug}/
SAMPLED    : data/financebench/sample_doc_labels.json (10 docs)
COST CACHE : results/financebench_single_cluster/llm/gpt54/one_shot/cost_profile/{question_slug}.json
COV CACHE  : results/financebench_single_cluster/llm/gpt54/one_shot/eval_individual/{question_slug}/
OUTPUT     : results/financebench_single_cluster/llm/gpt54/one_shot/selected_rules_agent/{question_slug}.json
TRACE      : results/financebench_single_cluster/llm/gpt54/one_shot/agent_trace/{question_slug}.jsonl

HARD CONSTRAINT (must be satisfied before you finish)
  Merge accuracy of your selected subset S must equal the merge accuracy of
  the full rule pool on every sampled doc the full pool solves correctly.
  Concretely: the tool below returns `missed_in_D_star: []` when this is met.

SOFT TARGETS (negotiate against each other)
  1. Minimise sum(avg_cost_ratio) across S.
  2. Maximise min cov(r) across r in S (broader rules generalise better).
  3. Keep |S| small. Prefer one broad rule over three narrow ones.

REASONABLE STOPPING SOFT-TARGET CRITERIA (subjective, optional):
  - min_cov(S) >= 0.4
  - sum_avg_cost_ratio(S) <= 0.5 * sum_avg_cost_ratio(full pool)
  - |S| <= 10
  Stop when accuracy matches AND any two of these three hold, or when budget
  is exhausted.

TOOLS YOU HAVE (invoke via the Bash tool, one per call)

  # Free (no LLM):
  python tools/list_rules.py --question-slug {question_slug}
  python tools/compute_cost.py --question-slug {question_slug} --rules <r1> <r2> ...
  python tools/compute_cost.py --question-slug {question_slug} --all
  python tools/compute_coverage.py --question-slug {question_slug} --rules <r1> <r2> ...
  python tools/compute_coverage.py --question-slug {question_slug} --all
  python tools/inspect_rule.py --question-slug {question_slug} --rule <name>

  # Paid (each call: ~20 gpt54 invocations on the 10 sampled docs):
  python tools/verify_accuracy.py \
      --question-slug {question_slug} \
      --question "{question}" \
      --rules <r1> <r2> ...

BUDGET: at most {budget} verify_accuracy calls. Use them sparingly.

LOOP (suggested; deviate if you have a better idea)
  1. list_rules to see the pool with one-line docstrings.
  2. compute_cost --all and compute_coverage --all to snapshot the pool.
  3. Propose an initial S using a cov/cost-descending heuristic.
     Rules of thumb:
       - rule names containing page numbers (page1, page_around_39) or
         month names (february, january) are often layout-specific. Prefer
         rules whose docstrings describe broader patterns.
       - cov(r) >= 0.5 and cost(r) <= 0.005 is a strong combination.
  4. verify_accuracy on S.
  5. If missed_in_D_star is nonempty:
       - inspect_rule on rules whose docstrings suggest they could cover those
         docs (e.g. for a phone-number question, rules that match cover-page
         tables vs MD&A sentences).
       - Add the most promising rule and re-verify.
  6. Once hard constraint holds, try drop-tests on the most expensive rule
     (recompute_cost --rules <without it> ; verify_accuracy --rules <without it>).
     Drop if accuracy preserved.
  7. Stop when soft targets feel reasonable or budget exhausted.

OUTPUT (the final step before exiting)

Write a JSON file to {output_path} with this exact schema (no extra fields):
{{
  "question":              "{question}",
  "question_slug":         "{question_slug}",
  "mode":                  "agentic",
  "model":                 "{model}",
  "selected_rules":        ["rule_a", "rule_b", "..."],
  "selected_avg_cost_ratio_sum": <float>,
  "min_cov":               <float>,
  "mean_cov":              <float>,
  "match_rate_on_sampled": <float>,
  "iterations":            <int>,    // number of edit cycles
  "verify_calls":          <int>,    // number of verify_accuracy invocations
  "rationale":             "<2-4 sentence explanation of decisions made>"
}}

Also append a per-step JSONL trace to {trace_path} (one object per tool call):
{{ "step": <int>, "tool": "<tool_name>", "args": "...", "result_summary": "..." }}

Then print to stdout a single line in this format:
  AGENTIC_SELECTION_DONE slug={question_slug} n_rules=<N> sum_cost=<F> min_cov=<F> match_rate=<F>

GUIDELINES
  - Prefer broad rules (cov >= 0.5) over narrow rules (cov < 0.3).
  - A rule that "wins" on cost but covers only one doc is almost always overfit;
    treat with suspicion unless it covers a doc no other rule reaches.
  - Read rule docstrings before adding. Names with page numbers or specific
    dates are usually layout-specific.
  - Do not refuse to finish. If you cannot satisfy the hard constraint within
    the budget, return the best S you have, set match_rate_on_sampled to the
    actual value, and explain in the rationale why.

You can call tools in parallel where independent (different rule subsets).
You may write small helper Python scripts via the Write tool if needed.

Work carefully, but do not over-think. The cost-effectiveness sort is a strong
prior; you should usually only need 2-4 verify_accuracy calls per question.
