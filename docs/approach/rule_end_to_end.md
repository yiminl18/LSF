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

Current checked-in results cover `court`, `nopv`, and `officeqa`. Gen model is `gpt54mini` throughout; apply model is `gpt54`.

#### Rule Generation summary

| Dataset | Split | Docs | Questions | Rules dir |
|---------|-------|-----:|----------:|-----------|
| Court | `all_docs_q1` | 294 | 1 | `rules/court/agentic_rule_full_data_gpt54mini/all_docs_q1` |
| Court | `all_docs` | 294 | 13 | `rules/court/agentic_rule_full_data_gpt54mini/all_docs` |
| NOPV | `all_docs` | 242 | 12 | `rules/nopv/agentic_rule_full_data_gpt54mini/all_docs` |
| OfficeQA | `all_docs` | 200 | 16 | `rules/officeqa/agentic_rule_full_data_gpt54mini/all_docs` |

Pricing for actual cost: $0.75/M non-cached input, $0.075/M cached input, $3.00/M output (`gpt54mini`). Cost ratio is `avg_cost_ratio_sample` — average retrieved-tokens / doc-tokens on the agent's working sample.

#### Rule Generation — Court (`all_docs_q1`, 294 docs, q01 only)

| Q | Question slug | Rules | Match rate | Latency | Input tokens | Cached | Output | Reasoning | Cost ratio | Actual cost |
|---|---------------|------:|-----------:|--------:|-------------:|-------:|-------:|----------:|-----------:|------------:|
| q01 | `what_isare_the_court_of_appeals_docket_numbers_for_this_case` | 1 | 1.000 | 202s | 1,258,334 | 1,198,336 | 29,092 | 22,873 | 0.0084 | $0.22 |

#### Rule Generation — Court (`all_docs`, 294 docs, 13 questions)

| Q | Question slug | Rule file(s) | Sample | Match rate | Latency | Input tokens | Output tokens | Cost ratio | Actual cost |
|---|---------------|-------------|-------:|-----------:|--------:|-------------:|--------------:|-----------:|------------:|
| q01 | `what_isare_the_court_of_appeals_docket_numbers_for_this_case` | `rule_court_of_appeals_docket_numbers.py` | 5 | 1.000 | 361s | 2,130,192 | 20,514 | 0.0120 | $0.28 |
| q02 | `what_isare_the_originating_district_court_docket_numbers_dc_` | `rule_originating_district_court_docket_numbers.py` | 5 | 1.000 | 563s | 4,153,764 | 36,985 | 0.0009 | $0.50 |
| q03 | `from_which_us_district_courts_did_this_appeal_originate` | `rule_us_district_court_origin.py` | 7 | 1.000 | 445s | 3,411,861 | 43,039 | 0.0096 | $0.48 |
| q04 | `which_district_judges_presided_over_the_case_below` | `rule_presiding_district_judge.py` | 7 | 1.000 | 631s | 4,726,199 | 52,847 | 0.0006 | $0.57 |
| q05 | `on_what_date_was_the_case_argued_or_submitted_on_the_briefs` | `rule_argument_submission_date_header.py` | 6 | 1.000 | 158s | 1,219,359 | 20,700 | 0.0037 | $0.21 |
| q06 | `in_what_city_was_the_case_argued_or_submitted` | `rule_city_after_argued_submitted_header.py`, `rule_city_from_oral_argument_order_phrase.py` | 6 | 1.000 | 211s | 1,231,621 | 20,117 | 0.0017 | $0.22 |
| q07 | `on_what_date_was_this_opinion_filed` | `rule_on_what_date_was_this_opinion_filed.py` | 3 | 1.000 | 351s | 2,821,975 | 46,685 | 0.0002 | $0.41 |
| q08 | `who_were_the_judges_on_the_panel_that_decided_this_case` | `rule_panel_judges_before_header.py` | 12 | 1.000 | 289s | 2,467,570 | 36,278 | 0.0060 | $0.37 |
| q09 | `which_judge_authored_the_majority_opinion` | `rule_opinion_by_judge_header.py` | 8 | 1.000 | 106s | 911,502 | 11,397 | 0.0005 | $0.15 |
| q10 | `what_legal_subject_matter_does_the_court_staff_summary_ident` | `rule_summary_topic.py` | 8 | 1.000 | 127s | 676,447 | 15,603 | 0.0005 | $0.15 |
| q11 | `who_is_the_firstlisted_attorney_representing_the_appellants` | `rule_first_listed_attorney_appellant.py` | 6 | 1.000 | 413s | 3,366,990 | 54,018 | 0.0005 | $0.46 |
| q12 | `who_is_the_firstlisted_attorney_representing_the_appellees` | `rule_first_listed_attorney_appellee.py` | 10 | 1.000 | 223s | 1,354,049 | 31,610 | 0.0005 | $0.24 |
| q13 | `what_was_the_panels_final_disposition_of_the_appeal` | `rule_final_disposition.py` | 3 | 1.000 | 501s | 6,623,814 | 65,477 | 0.0007 | $0.80 |

#### Rule Generation — NOPV (`all_docs`, 242 docs)

| Q | Question slug | Rule file | Sample | Match rate | Latency | Input tokens | Output tokens | Cost ratio | Actual cost |
|---|---------------|-----------|-------:|-----------:|--------:|-------------:|--------------:|-----------:|------------:|
| q01 | `what_is_the_legal_name_of_the_pipeline_operator_to_whom_this` | `rule_notice_addressee_operator_name.py` | 7 | 1.000 | 180s | 1,582,192 | 25,214 | 0.0018 | $0.25 |
| q02 | `what_is_the_cpf_case_number_assigned_to_this_enforcement_act` | `rule_cpf_case_number.py` | 4 | 1.000 | 704s | 3,544,868 | 47,957 | 0.0066 | $0.52 |
| q03 | `which_phmsa_region_office_eg_eastern_southern_central_wester` | `rule_phmsa_region_office_signature.py` | 5 | 1.000 | 300s | 2,041,432 | 20,429 | 0.0049 | $0.28 |
| q04 | `on_what_date_was_this_notice_of_probable_violation_issued` | `rule_notice_issuance_date.py` | 7 | 0.714 | 1010s | 6,090,298 | 58,295 | 0.0046 | $0.82 |
| q05 | `on_what_date_or_date_range_did_phmsa_conduct_the_inspection_` | `rule_phmsa_inspection_date_range.py` | 6 | 1.000 | 254s | 3,509,938 | 35,941 | 0.0039 | $0.45 |
| q06 | `in_which_us_state_or_states_is_the_pipeline_facility_address` | `rule_opening_location_sentence.py` | 6 | 1.000 | 607s | 4,905,969 | 34,447 | 0.0233 | $0.59 |
| q07 | `under_which_49_cfr_part_is_the_operators_regulated_activity_` | `rule_operator_regulated_part.py` | 8 | 1.000 | 247s | 2,019,692 | 24,934 | 0.0080 | $0.35 |
| q08 | `how_many_distinct_alleged_violations_are_enumerated_in_the_b` | `rule_alleged_violations_body_section.py` | 5 | 1.000 | 493s | 4,277,253 | 50,509 | 0.4637 | $0.63 |
| q09 | `list_all_49_cfr_section_numbers_cited_as_the_basis_for_alleg` | `rule_basis_section_headings.py` | 4 | 0.750 | 800s | 9,721,713 | 93,351 | 0.0246 | $1.10 |
| q10 | `how_many_distinct_corrective_action_items_are_listed_in_the_` | `rule_proposed_compliance_order_corrective_items.py` | 7 | 0.143 | 994s | 11,560,212 | 112,695 | 0.0003 | $1.45 |
| q11 | `what_is_the_longest_deadline_in_days_measured_from_the_final` | `rule_proposed_compliance_order_deadlines.py` | 5 | 1.000 | 216s | 1,681,737 | 26,231 | 0.0461 | $0.26 |
| q12 | `who_signed_this_notice_on_behalf_of_phmsa_regional_director_` | `rule_phmsa_notice_signature_block.py` | 5 | 1.000 | 215s | 2,818,953 | 27,475 | 0.0016 | $0.37 |

#### Rule Generation — OfficeQA (`all_docs`, 200 docs)

| Q | Question slug | Rule file(s) | Sample | Match rate | Latency | Input tokens | Output tokens | Cost ratio | Actual cost |
|---|---------------|-------------|-------:|-----------:|--------:|-------------:|--------------:|-----------:|------------:|
| q01 | `what_quarter_and_year_does_this_treasury_bulletin_cover` | `rule_treasury_bulletin_cover_period.py` | 5 | 0.800 | 290s | 1,750,525 | 41,918 | 0.2002 | $0.30 |
| q02 | `what_was_the_annualized_real_gdp_growth_rate_for_the_most_re` | `rule_legacy_real_gross_domestic_product.py`, `rule_recent_real_gdp_growth.py` | 5 | 1.000 | 144s | 790,500 | 21,153 | 0.0032 | $0.15 |
| q03 | `what_was_the_unemployment_rate_at_the_most_recent_month_repo` | `rule_unemployment_latest_month_sentence.py` | 5 | 1.000 | 514s | 2,936,970 | 36,714 | 0.0003 | $0.39 |
| q04 | `what_was_the_average_monthly_nonfarm_payroll_job_growth_so_f` | `rule_payroll_monthly_gain_first_quarter.py`, `rule_payroll_year_to_date_average.py` | 3 | 1.000 | 191s | 1,001,724 | 21,923 | 0.0007 | $0.20 |
| q05 | `what_was_the_university_of_michigan_reuters_consumer_sentime` | `rule_reuters_michigan_consumer_sentiment_latest_reading.py`, `rule_umich_consumer_sentiment_latest_reading.py` | 5 | 1.000 | 166s | 1,101,187 | 21,026 | 0.0004 | $0.18 |
| q06 | `what_was_the_federal_budget_deficit_for_the_most_recent_comp` | `rule_budget_year_percent.py`, `rule_federal_budget_heading_window.py`, `rule_fiscal_year_line_fallback.py`, `rule_full_text_budget_sentence.py`, `rule_specific_1991_gnp_share.py` | 4 | 1.000 | 600s | 4,938,939 | 44,209 | 0.0901 | $0.84 |
| q07 | `what_were_total_federal_receipts_for_the_fiscal_year_to_date` | `rule_ffo4_total_receipts_table.py`, `rule_total_federal_receipts_narrative.py` | 5 | 1.000 | 435s | 5,158,105 | 47,821 | 0.0001 | $0.68 |
| q08 | `what_was_the_total_surplus_or_deficit_for_the_fiscal_year_to` | `rule_ffo1_total_surplus_or_deficit_table.py` | 4 | 1.000 | 379s | 3,402,987 | 33,583 | 0.0001 | $0.42 |
| q09 | `what_were_net_individual_income_tax_receipts_for_the_most_re` | `rule_individual_income_tax_receipts_quarter.py` | 5 | 1.000 | 184s | 925,017 | 18,614 | 0.0000 | $0.16 |
| q10 | `what_was_the_total_gross_federal_debt_outstanding_at_the_end` | `rule_fd1_latest_gross_federal_debt.py` | 5 | 1.000 | 677s | 6,147,789 | 71,334 | 0.0002 | $0.91 |
| q11 | `what_was_the_total_federal_debt_held_by_the_public_at_the_en` | `rule_fd1_latest_federal_debt_held_by_public.py` | 5 | 1.000 | 217s | 1,465,383 | 26,298 | 0.0002 | $0.23 |
| q12 | `is_the_statutory_debt_limit_currently_suspended_or_set_at_a_` | `rule_debt_limit_suspension_note.py`, `rule_fd6_latest_statutory_debt_limit.py` | 3 | 1.000 | 210s | 2,235,001 | 24,690 | 0.0111 | $0.32 |
| q13 | `what_was_the_average_length_in_months_of_marketable_interest` | `rule_fd5_avg_length_recent_fiscal_year.py` | 3 | 1.000 | 184s | 1,508,652 | 19,112 | 0.0067 | $0.23 |
| q14 | `what_was_the_per_capita_amount_of_currency_and_coin_in_circu` | `rule_ms1_latest_per_capita_currency_and_coin.py`, `rule_uscc2_latest_per_capita_currency_and_coin.py` | 2 | 1.000 | 223s | 1,363,827 | 24,452 | 0.0001 | $0.25 |
| q15 | `what_was_the_exchange_stabilization_fund_total_assets_in_tho` | `rule_esf_total_assets.py` | 4 | 1.000 | 255s | 2,165,607 | 26,773 | 0.0014 | $0.30 |
| q16 | `what_was_the_most_recent_weekly_exchange_rate_of_canadian_do` | `rule_canadian_dollar_weekly_exchange_rate.py` | 4 | 1.000 | 483s | 4,936,726 | 54,518 | 0.0000 | $0.61 |

#### Rule Application — Court (`all_docs`, 294 docs, `rule_apply_merge` + `gpt54`)

| Split | Q | Question slug | Accuracy | Cost ratio | Latency | Note |
|-------|---|---------------|--------:|-----------:|--------:|------|
| `all_docs_q1` | q01 | `what_isare_the_court_of_appeals_docket_numbers_for_this_case` | **0.9864** | 0.0332 | 0.76s | |
| `all_docs` | q01 | `what_isare_the_court_of_appeals_docket_numbers_for_this_case` | 0.9422 | 0.0328 | 0.85s | |
| `all_docs` | q02 | `what_isare_the_originating_district_court_docket_numbers_dc_` | 0.4592 | 0.0349 | 0.89s | |
| `all_docs` | q03 | `from_which_us_district_courts_did_this_appeal_originate` | 0.7687 | 0.0368 | 0.81s | |
| `all_docs` | q04 | `which_district_judges_presided_over_the_case_below` | 0.6905 | 0.0303 | 0.85s | Py3.9 fix applied |
| `all_docs` | q05 | `on_what_date_was_the_case_argued_or_submitted_on_the_briefs` | 0.7925 | 0.0306 | 0.83s | Py3.9 fix applied |
| `all_docs` | q06 | `in_what_city_was_the_case_argued_or_submitted` | 0.7109 | 0.0291 | 0.90s | |
| `all_docs` | q07 | `on_what_date_was_this_opinion_filed` | **1.0000** | 0.0304 | 0.97s | |
| `all_docs` | q08 | `who_were_the_judges_on_the_panel_that_decided_this_case` | 0.9082 | 0.0330 | 1.06s | |
| `all_docs` | q09 | `which_judge_authored_the_majority_opinion` | 0.7619 | 0.0281 | 0.94s | |
| `all_docs` | q10 | `what_legal_subject_matter_does_the_court_staff_summary_ident` | 0.8197 | 0.0317 | 0.90s | |
| `all_docs` | q11 | `who_is_the_firstlisted_attorney_representing_the_appellants` | 0.6905 | 0.0295 | 0.83s | Py3.9 fix applied |
| `all_docs` | q12 | `who_is_the_firstlisted_attorney_representing_the_appellees` | 0.5068 | 0.0298 | 0.98s | |
| `all_docs` | q13 | `what_was_the_panels_final_disposition_of_the_appeal` | 0.5000 | 0.0353 | 0.81s | Py3.9 fix applied |

Notes:
- q01 differs between splits (0.9864 vs 0.9422): two agent runs produced different rule implementations.
- q02/q03 lower accuracy: multi-value extraction (multiple docket numbers, multiple district courts).
- q04/q05/q11/q13 previously showed 0.000 due to the Python 3.9 annotation bug; all four have been re-run with the fix applied.
- q07 achieves perfect accuracy — "Filed \<date\>" header is fully consistent across all docs.

#### Rule Application — NOPV (`all_docs`, 242 docs, `rule_apply_merge` + `gpt54`)

| Q | Question slug | Accuracy | Cost ratio | Latency |
|---|---------------|--------:|-----------:|--------:|
| q01 | `what_is_the_legal_name_of_the_pipeline_operator_to_whom_this` | 0.7355 | 0.0372 | 0.82s |
| q02 | `what_is_the_cpf_case_number_assigned_to_this_enforcement_act` | **0.9835** | 0.0409 | 0.89s |
| q03 | `which_phmsa_region_office_eg_eastern_southern_central_wester` | 0.9380 | 0.0449 | 0.74s |
| q04 | `on_what_date_was_this_notice_of_probable_violation_issued` | **0.9793** | 0.0366 | 0.78s |
| q05 | `on_what_date_or_date_range_did_phmsa_conduct_the_inspection_` | 0.8636 | 0.0409 | 0.85s |
| q06 | `in_which_us_state_or_states_is_the_pipeline_facility_address` | 0.7686 | 0.0711 | 0.82s |
| q07 | `under_which_49_cfr_part_is_the_operators_regulated_activity_` | 0.1653 | 0.0482 | 0.75s |
| q08 | `how_many_distinct_alleged_violations_are_enumerated_in_the_b` | 0.5041 | 0.0380 | 0.83s |
| q09 | `list_all_49_cfr_section_numbers_cited_as_the_basis_for_alleg` | 0.4752 | 0.0472 | 0.92s |
| q10 | `how_many_distinct_corrective_action_items_are_listed_in_the_` | 0.0000 | 0.0351 | 0.77s |
| q11 | `what_is_the_longest_deadline_in_days_measured_from_the_final` | 0.8678 | 0.0787 | 0.74s |
| q12 | `who_signed_this_notice_on_behalf_of_phmsa_regional_director_` | 0.8678 | 0.0386 | 0.83s |

Notes:
- q02–q05, q11–q12 are header/signature fields with high consistency → 0.86–0.98 accuracy.
- q10 (corrective action items count) has 0% — the agent's working sample match rate was only 0.14 during generation, indicating the rule was already weak.
- q07–q09 are multi-value or hierarchical fields (CFR parts, section lists, violation counts) → moderate accuracy.

#### Rule Application — OfficeQA (`all_docs`, 200 docs, `rule_apply_merge` + `gpt54`)

| Q | Question slug | Accuracy | Cost ratio | Latency |
|---|---------------|--------:|-----------:|--------:|
| q01 | `what_quarter_and_year_does_this_treasury_bulletin_cover` | 0.6400 | 0.1211 | 0.90s |
| q02 | `what_was_the_annualized_real_gdp_growth_rate_for_the_most_re` | 0.5500 | 0.1279 | 0.80s |
| q03 | `what_was_the_unemployment_rate_at_the_most_recent_month_repo` | 0.5150 | 0.1176 | 0.86s |
| q04 | `what_was_the_average_monthly_nonfarm_payroll_job_growth_so_f` | 0.1950 | 0.1277 | 0.82s |
| q05 | `what_was_the_university_of_michigan_reuters_consumer_sentime` | 0.0400 | 0.1236 | 0.76s |
| q06 | `what_was_the_federal_budget_deficit_for_the_most_recent_comp` | 0.5550 | 0.1934 | 1.09s |
| q07 | `what_were_total_federal_receipts_for_the_fiscal_year_to_date` | 0.3150 | 0.1236 | 0.73s |
| q08 | `what_was_the_total_surplus_or_deficit_for_the_fiscal_year_to` | 0.4500 | 0.1261 | 0.76s |
| q09 | `what_were_net_individual_income_tax_receipts_for_the_most_re` | 0.4250 | 0.1261 | 0.77s |
| q10 | `what_was_the_total_gross_federal_debt_outstanding_at_the_end` | 0.2350 | 0.1311 | 0.83s |
| q11 | `what_was_the_total_federal_debt_held_by_the_public_at_the_en` | 0.4200 | 0.1337 | 0.80s |
| q12 | `is_the_statutory_debt_limit_currently_suspended_or_set_at_a_` | 0.3400 | 0.1217 | 0.81s |
| q13 | `what_was_the_average_length_in_months_of_marketable_interest` | 0.3650 | 0.1384 | 0.83s |
| q14 | `what_was_the_per_capita_amount_of_currency_and_coin_in_circu` | 0.0300 | 0.1298 | 0.77s |
| q15 | `what_was_the_exchange_stabilization_fund_total_assets_in_tho` | 0.5350 | 0.1221 | 0.79s |
| q16 | `what_was_the_most_recent_weekly_exchange_rate_of_canadian_do` | 0.3700 | 0.1248 | 0.76s |

Notes:
- OfficeQA accuracy is moderate across the board (0.03–0.64), lower than Court or NOPV. Treasury Bulletins span many decades with varying table layouts, making rules fragile across the full 200-doc corpus even when they work on a small sample.
- Cost ratios (~0.12) are much higher than Court (~0.03) or NOPV (~0.04), reflecting that Treasury Bulletin docs are longer and the rules tend to retrieve larger text windows.
- q05 (consumer sentiment, 0.04) and q14 (per-capita currency, 0.03) are near-zero — these values appear in very layout-specific table positions that the generated rules do not generalize across.
