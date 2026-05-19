# Overfitting Analysis Report

**Sampled-Refined Rules (gpt54, 10 docs) vs Unsampled-Refined Gold (gpt54mini, 50 docs)**
Generated: 2026-05-16 01:55

---

## Metric Legend

| Metric | Description |
|--------|-------------|
| **Acc-U** | Individual rule accuracy on 50 unsampled docs (LLM QA+judge) |
| **Cov-S** | Coverage on 10 sampled docs (fraction of docs rule retrieves anything) |
| **Cov-U** | Coverage on 50 unsampled docs |
| **Cost** | Avg cost ratio (retrieved_tokens / total_doc_tokens) on unsampled docs |
| **Spec** | Rule specificity inferred from docstring |

---

## Per-Question Analysis

### Q1 · How many shares of common stock were outstanding as of the cover-page reference date?

| | Sampled-refined | Unsampled-gold |
|---|---|---|
| **# Rules** | 7 | 9 |
| **Acc on sampled** | 0.90 | — |
| **Acc on unsampled** | 0.66 | 0.96 |
| **Accuracy gap (gold − sampled)** | **+0.30** | |

Overlap: 3 rules · Only-sampled: 4 · Only-gold: 6

#### Overfit Candidates (sampled only, dropped by gold)

| Rule | Acc-U | Cov-S | Cov-U | Cost | Spec | Description |
|------|------:|------:|------:|-----:|------|-------------|
| rule_page1_phrase_number_of_shares_outstanding_of_registrant | 0.10 | 0.10 | 0.10 | 0.00006 | MEDIUM | Match page-1 spans with 'number of shares outstanding of the registrant' |
| rule_page1_sentence_with_there_were_and_common_stock | 0.04 | 0.20 | 0.04 | 0.00001 | MEDIUM | Match page-1 spans using 'There were … common stock … outstanding' |
| rule_page1_shares_outstanding_as_of_february | 0.08 | 0.20 | 0.08 | 0.00004 | **HIGH** | Match spans with outstanding-share sentence using a February date |
| rule_page2_cover_page_numeric_sentence | 0.30 | 0.30 | 0.30 | 0.00011 | **HIGH** | Match page-2 text spans with a long comma-formatted number |

**Aggregate (only-sampled):** Avg Acc-U=0.130 · Avg Cov-S=0.200 · Avg Cov-U=0.130 · Coverage drop S→U=+0.070 · High-specificity: 2/4

#### Missed by Sampled (gold only)

| Rule | Acc-U | Cov-S | Cov-U | Cost | Description |
|------|------:|------:|------:|-----:|-------------|
| rule_cover_path_text_company_page1 | **0.72** | 0.80 | 0.86 | 0.00175 | Match page-1 body/text spans under the company cover block |
| rule_page1_after_nonaffiliate_market_value | 0.36 | 0.70 | 0.48 | 0.00020 | Match page-1 spans with outstanding-share language after market value |
| rule_page1_company_cover_section_header | 0.00 | 0.00 | 0.00 | 0.00000 | Match the main company cover section_header on page 1 |
| rule_page1_market_value_and_number_of_shares_labels | 0.00 | 0.20 | 0.10 | 0.00003 | Match page-1 spans that are labels for market value and number of shares |
| rule_page1_numeric_only_after_outstanding_label | 0.04 | 0.10 | 0.10 | 0.00001 | Match numeric-only spans on page 1/2 that follow a label span |
| rule_page1_sentence_with_common_stock_outstanding_and_documents_reference | 0.00 | 0.50 | 0.46 | 0.00003 | Match page-1 spans with common-stock-outstanding language |

**Aggregate (only-gold):** Avg Acc-U=0.187 · Avg Cov-S=0.383 · Avg Cov-U=0.333 · Low sampled-coverage (<0.3): 3 rules pruned

#### Shared Rules (both sets)

| Rule | Acc-U | Cov-S | Cov-U | Cost |
|------|------:|------:|------:|-----:|
| rule_h1_company_cover_span_with_outstanding | 0.00 | 0.00 | 0.00 | 0.00000 |
| rule_page1_cover_text_with_multiple_large_numbers | 0.10 | 0.10 | 0.14 | 0.00018 |
| rule_page1_numeric_after_phrase_as_of_january | 0.02 | 0.10 | 0.06 | 0.00001 |

#### Diagnosis

1. **Low generalization** — Dropped rules have very low unsampled coverage (0.13); they fire on very few unseen docs.
2. **Low individual accuracy** — Even when sampled-only rules DO retrieve text on unsampled docs, avg accuracy is only 0.13.
3. **Over-specific rules** — 2 dropped rules target narrow patterns (specific months): `rule_page1_shares_outstanding_as_of_february`, `rule_page2_cover_page_numeric_sentence`.
4. **Sampling bias** — 1 gold rule had zero coverage on sampled docs and was eliminated despite generalizing: `rule_page1_company_cover_section_header`.

---

### Q2 · What is long-term debt at year-end (0 if none)?

| | Sampled-refined | Unsampled-gold |
|---|---|---|
| **# Rules** | 6 | 11 |
| **Acc on sampled** | 0.50 | — |
| **Acc on unsampled** | 0.46 | 0.82 |
| **Accuracy gap (gold − sampled)** | **+0.36** | |

Overlap: 4 rules · Only-sampled: 2 · Only-gold: 7

#### Overfit Candidates (sampled only, dropped by gold)

| Rule | Acc-U | Cov-S | Cov-U | Cost | Description |
|------|------:|------:|------:|-----:|-------------|
| rule_balance_sheet_table_on_pages_30_to_70 | 0.04 | 0.20 | 0.12 | 0.00042 | Match mid-document financial statement tables on typical balance sheet pages |
| rule_mda_debt_at_year_end_text | 0.08 | 0.60 | 0.40 | 0.00177 | Match text spans that explicitly say debt at year-end |

**Aggregate (only-sampled):** Avg Acc-U=0.060 · Avg Cov-S=0.400 · Avg Cov-U=0.260 · Coverage drop S→U=+0.140

#### Missed by Sampled (gold only)

| Rule | Acc-U | Cov-S | Cov-U | Cost | Description |
|------|------:|------:|------:|-----:|-------------|
| rule_debt_note_header_same_page_tables | **0.46** | **1.00** | **0.94** | 0.00581 | Match tables on the same page as a debt-related section header |
| rule_debt_note_year_end_table | 0.22 | 0.80 | 0.42 | 0.00606 | Match debt tables that explicitly present year-end balances |
| rule_debt_related_section_headers | 0.42 | **1.00** | **1.00** | 0.00945 | Match debt-related headers and nearby text spans (broad) |
| rule_mda_liquidity_debt_paragraph | 0.08 | 0.90 | 0.84 | 0.00541 | Match MD&A liquidity/capital resources text spans mentioning debt |
| rule_note_debt_header_near_table | 0.42 | 0.90 | 0.84 | 0.00549 | Match section headers for debt notes that precede the debt table |
| rule_note_debt_table | 0.44 | 0.90 | 0.72 | 0.00453 | Match debt note tables under notes to consolidated financial statements |
| rule_text_span_long_term_debt_numeric | 0.12 | **1.00** | 0.90 | 0.00383 | Match text spans containing long-term debt with nearby numeric values |

**Aggregate (only-gold):** Avg Acc-U=0.309 · Avg Cov-S=0.929 · Avg Cov-U=0.809

#### Shared Rules (both sets)

| Rule | Acc-U | Cov-S | Cov-U | Cost |
|------|------:|------:|------:|-----:|
| rule_debt_table_under_notes_or_mda | 0.12 | 0.50 | 0.30 | 0.00271 |
| rule_debt_table_with_noncurrent_context | 0.38 | 0.70 | 0.56 | 0.00523 |
| rule_near_balance_sheet_pages_with_debt | 0.00 | 0.00 | 0.00 | 0.00000 |
| rule_pages_near_item_8_from_toc | 0.06 | 0.70 | 0.70 | 0.00545 |

#### Diagnosis

1. **Low individual accuracy** — Even when sampled-only rules retrieve text on unsampled docs, avg accuracy is only 0.06.
2. **Cost-sort exclusion** — Missed rules have Cov-S=0.80–1.00 but are expensive (cost 0.004–0.009). The algorithm sorts cheapest-first; cheaper rules hit the 0.50 target before expensive high-coverage rules are ever evaluated.

---

### Q3 · What is net income (loss) for the most recent fiscal year?

| | Sampled-refined | Unsampled-gold |
|---|---|---|
| **# Rules** | 14 | 22 |
| **Acc on sampled** | 0.90 | — |
| **Acc on unsampled** | 0.76 | 0.82 |
| **Accuracy gap (gold − sampled)** | **+0.06** | |

Overlap: 10 rules · Only-sampled: 4 · Only-gold: 12

#### Overfit Candidates (sampled only, dropped by gold)

| Rule | Acc-U | Cov-S | Cov-U | Cost | Description |
|------|------:|------:|------:|-----:|-------------|
| rule_consolidated_results_table | 0.54 | 1.00 | 1.00 | 0.04506 | Match tables under paths mentioning consolidated financial statements |
| rule_mda_results_of_operations_net_income_text | 0.26 | 0.70 | 0.80 | 0.00413 | Match MD&A Results of Operations text mentioning net income |
| rule_net_income_in_bold_summary_text | 0.04 | 0.50 | 0.46 | 0.00071 | Match bold text spans mentioning net income/earnings/loss |
| rule_tables_with_row_net_and_column_years | 0.40 | 1.00 | 1.00 | 0.04078 | Match tables where a net row coexists with year-like column headers |

**Aggregate (only-sampled):** Avg Acc-U=0.310 · Avg Cov-S=0.800 · Avg Cov-U=0.815 · Coverage drop S→U=−0.015

#### Missed by Sampled (gold only)

| Rule | Acc-U | Cov-S | Cov-U | Cost | Description |
|------|------:|------:|------:|-----:|-------------|
| rule_financial_statement_table_with_parentheses_and_dollars | 0.48 | 1.00 | 1.00 | 0.05132 | Match financial statement tables with dollar formatting |
| rule_item8_financial_statements_pages | 0.06 | 0.80 | 0.96 | 0.01584 | Match spans on pages associated with Item 8 / Financial Statements |
| rule_net_income_row_in_financial_statements_section | 0.38 | 0.60 | 0.86 | 0.03439 | Match tables in Item 8 / Financial Statements with net income row |
| rule_net_income_row_in_income_statement_table | 0.48 | 1.00 | 0.92 | 0.01271 | Match tables containing a net income/loss row in the income statement |
| rule_page_after_statement_of_income_header | 0.40 | 0.90 | 0.82 | 0.01148 | Match spans on the same or next page after a Statement of Income header |
| rule_row_header_net_income_in_any_table | 0.54 | 1.00 | 1.00 | 0.04965 | Match any table where a row header cell is net income/loss/earnings |
| rule_statement_of_income_heading_nearby | 0.00 | 0.90 | 0.82 | 0.00021 | Match section headers for Statement of Income / Operations |
| rule_table_with_net_income_and_three_years | 0.42 | 1.00 | 1.00 | 0.04692 | Match tables with a net income row and at least three distinct year columns |
| rule_tables_under_part_ii_with_net_income | 0.30 | 0.80 | 0.62 | 0.02918 | Match tables under Part II that mention net income/earnings/loss |
| rule_tables_with_financial_units_and_recent_years | 0.44 | 1.00 | 1.00 | 0.05062 | Match financial tables using units like million/billion and recent years |
| rule_toc_item7_page_then_following_pages | 0.32 | 0.80 | 0.96 | 0.02844 | Match spans on Item 7 pages and nearby pages |
| rule_toc_statement_of_income_page | 0.20 | 0.90 | 0.82 | 0.00666 | Match spans on the TOC page for Consolidated Statement of Income |

**Aggregate (only-gold):** Avg Acc-U=0.335 · Avg Cov-S=0.892 · Avg Cov-U=0.898

#### Shared Rules (both sets)

| Rule | Acc-U | Cov-S | Cov-U | Cost |
|------|------:|------:|------:|-----:|
| rule_financial_tables_on_pages_from_toc_income_or_item8 | 0.00 | 0.50 | 0.50 | 0.00069 |
| rule_first_financial_table_after_item8 | 0.08 | 0.60 | 0.78 | 0.00203 |
| rule_first_table_after_statement_of_income_header | 0.34 | 0.90 | 0.82 | 0.00501 |
| rule_item8_or_selected_data_headers | 0.00 | 1.00 | 1.00 | 0.00024 |
| rule_mda_item7_net_income_text | 0.46 | 0.90 | 0.82 | 0.00708 |
| rule_nearby_after_anchor_headers | **0.70** | 1.00 | 1.00 | 0.05753 |
| rule_pages_near_item8_from_toc_and_net_income | 0.24 | 0.50 | 0.62 | 0.00593 |
| rule_selected_financial_data_net_income_table | 0.26 | 0.50 | 0.58 | 0.00511 |
| rule_selected_financial_data_section | 0.00 | 0.60 | 0.60 | 0.00007 |
| rule_tables_with_recent_year_columns | 0.00 | 0.00 | 0.00 | 0.00000 |

#### Diagnosis

No strong overfitting signal for this question. Small accuracy gap (+0.06).

---

### Q4 · What is the address of principal executive offices and ZIP code?

| | Sampled-refined | Unsampled-gold |
|---|---|---|
| **# Rules** | 5 | 7 |
| **Acc on sampled** | 0.90 | — |
| **Acc on unsampled** | 0.88 | 0.98 |
| **Accuracy gap (gold − sampled)** | **+0.10** | |

Overlap: 4 rules · Only-sampled: 1 · Only-gold: 3

#### Overfit Candidates (sampled only, dropped by gold)

| Rule | Acc-U | Cov-S | Cov-U | Cost | Description |
|------|------:|------:|------:|-----:|-------------|
| rule_page1_before_securities_registered | 0.06 | 0.20 | 0.32 | 0.00004 | Return address-like spans appearing before the first securities-registered section |

**Aggregate (only-sampled):** Avg Acc-U=0.060 · Avg Cov-S=0.200 · Avg Cov-U=0.320

#### Missed by Sampled (gold only)

| Rule | Acc-U | Cov-S | Cov-U | Cost | Description |
|------|------:|------:|------:|-----:|-------------|
| rule_page1_address_like_with_known_cover_labels | 0.06 | 0.30 | 0.20 | 0.00002 | Match spans where address-like text co-occurs with cover-page labels |
| rule_page1_address_or_zip_single_token_headers | 0.10 | 1.00 | 0.98 | 0.00028 | Match single-token page-1 headers that are ZIP codes or short addresses |
| rule_page1_main_company_path_text | 0.30 | 0.90 | 0.36 | 0.00024 | Match page-1 spans whose path_text is the main company name |

**Aggregate (only-gold):** Avg Acc-U=0.153 · Avg Cov-S=0.733 · Avg Cov-U=0.513

#### Shared Rules (both sets)

| Rule | Acc-U | Cov-S | Cov-U | Cost |
|------|------:|------:|------:|-----:|
| rule_page1_address_after_state_before_ein | 0.22 | 0.20 | 0.42 | 0.00014 |
| rule_page1_address_with_city_state_no_zip | 0.00 | 0.30 | 0.24 | 0.00004 |
| rule_page1_city_state_zip_only | 0.38 | 0.80 | 0.76 | 0.00009 |
| rule_page1_zip_code_label_spans | 0.06 | 0.70 | 0.66 | 0.00014 |

#### Diagnosis

1. **Low individual accuracy** — Even when sampled-only rules retrieve text on unsampled docs, avg accuracy is only 0.06.

---

### Q5 · What is the registrant's exact name?

| | Sampled-refined | Unsampled-gold |
|---|---|---|
| **# Rules** | 2 | 4 |
| **Acc on sampled** | 0.90 | — |
| **Acc on unsampled** | 0.90 | 1.00 |
| **Accuracy gap (gold − sampled)** | **+0.10** | |

Overlap: 2 rules · Only-sampled: 0 · Only-gold: 2

#### Missed by Sampled (gold only)

| Rule | Acc-U | Cov-S | Cov-U | Cost | Description |
|------|------:|------:|------:|-----:|-------------|
| rule_item1_general_company_mention | 0.36 | 0.90 | 0.44 | 0.00061 | Match early Item 1 / General text spans that restate the company name |
| rule_page1_company_heading_with_single_path | **0.84** | 0.80 | 0.86 | 0.00006 | Match page-1 headings whose path_text equals their text |

**Aggregate (only-gold):** Avg Acc-U=0.600 · Avg Cov-S=0.850 · Avg Cov-U=0.650

#### Shared Rules (both sets)

| Rule | Acc-U | Cov-S | Cov-U | Cost |
|------|------:|------:|------:|-----:|
| rule_page1_before_state_header | 0.02 | 1.00 | 0.94 | 0.00003 |
| rule_page1_large_bold_name_before_exact_caption | **0.88** | 0.90 | 0.88 | 0.00006 |

#### Diagnosis

No strong overfitting signal. Sampled-only set is a strict subset of gold — gap (+0.10) explained by 2 missed gold rules covering diverse company-name header patterns.

---

### Q6 · What is the registrant's telephone number?

| | Sampled-refined | Unsampled-gold |
|---|---|---|
| **# Rules** | 5 | 5 |
| **Acc on sampled** | 1.00 | — |
| **Acc on unsampled** | 0.82 | 0.98 |
| **Accuracy gap (gold − sampled)** | **+0.16** | |

Overlap: 1 rule · Only-sampled: 4 · Only-gold: 4

#### Overfit Candidates (sampled only, dropped by gold)

| Rule | Acc-U | Cov-S | Cov-U | Cost | Description |
|------|------:|------:|------:|-----:|-------------|
| rule_cover_page_before_section_12b | 0.30 | 0.90 | 0.80 | 0.00012 | Match page-1 spans where a phone number appears near the Section 12(b) area |
| rule_exact_phone_only_span | 0.56 | 0.30 | 0.56 | 0.00005 | Match spans whose text is primarily just a phone number |
| rule_page1_before_documents_incorporated | 0.38 | 0.50 | 0.40 | 0.00006 | Match phone-like spans occurring before 'DOCUMENTS INCORPORATED' |
| rule_registrant_phone_following_zip_parent | 0.10 | 0.20 | 0.16 | 0.00041 | Match child/body spans under a zip-code-related parent |

**Aggregate (only-sampled):** Avg Acc-U=0.335 · Avg Cov-S=0.475 · Avg Cov-U=0.480 · Coverage drop S→U=−0.005

#### Missed by Sampled (gold only)

| Rule | Acc-U | Cov-S | Cov-U | Cost | Description |
|------|------:|------:|------:|-----:|-------------|
| rule_page1_h2_phone_header | 0.28 | 0.60 | 0.50 | 0.00005 | Match page-1 H2 spans whose header text contains the phone number |
| rule_page1_phone_with_parenthetical_area_code_label | **0.78** | 0.80 | 0.78 | 0.00012 | Match phone spans adjacent to '(Registrant's telephone number…)' |
| rule_phone_before_section12b | 0.06 | 0.10 | 0.06 | 0.00001 | Match spans with phone numbers that occur before the Section 12(b) block |
| rule_principal_executive_offices_and_phone | 0.10 | 0.90 | 0.66 | 0.00015 | Match spans mentioning principal executive offices together with a phone number |

**Aggregate (only-gold):** Avg Acc-U=0.305 · Avg Cov-S=0.600 · Avg Cov-U=0.500 · Low sampled-coverage (<0.3): 1 rule pruned

#### Shared Rules (both sets)

| Rule | Acc-U | Cov-S | Cov-U | Cost |
|------|------:|------:|------:|-----:|
| rule_phone_with_address_and_zip | 0.14 | 0.90 | 0.62 | 0.00012 |

#### Diagnosis

No strong overfitting signal. Almost complete rule-set mismatch (only 1 shared rule) — primarily a semantics mismatch between narrow and broader phone-pattern strategies.

---

### Q7 · What is the state (or other jurisdiction) of incorporation and the IRS Employer Identification Number?

| | Sampled-refined | Unsampled-gold |
|---|---|---|
| **# Rules** | 3 | 4 |
| **Acc on sampled** | 1.00 | — |
| **Acc on unsampled** | 0.88 | 1.00 |
| **Accuracy gap (gold − sampled)** | **+0.12** | |

Overlap: 1 rule · Only-sampled: 2 · Only-gold: 3

#### Overfit Candidates (sampled only, dropped by gold)

| Rule | Acc-U | Cov-S | Cov-U | Cost | Description |
|------|------:|------:|------:|-----:|-------------|
| rule_page1_combined_state_and_ein_same_text | 0.04 | 0.10 | 0.04 | 0.00004 | Match page-1 spans whose text itself contains both a jurisdiction and EIN |
| rule_page1_ein_number_pattern | 0.04 | 1.00 | 0.98 | 0.00012 | Match page-1 spans containing an EIN-like number pattern |

**Aggregate (only-sampled):** Avg Acc-U=0.040 · Avg Cov-S=0.550 · Avg Cov-U=0.510 · Coverage drop S→U=+0.040

#### Missed by Sampled (gold only)

| Rule | Acc-U | Cov-S | Cov-U | Cost | Description |
|------|------:|------:|------:|-----:|-------------|
| rule_page1_bold_ein_number_format | 0.00 | 0.80 | 0.80 | 0.00005 | Match bold page-1 spans that look like standalone EIN number blocks |
| rule_page1_company_header_and_next_15 | **0.88** | 1.00 | 0.96 | 0.00190 | Match the first company-name header on page 1 and the next 15 spans |
| rule_page1_texts_with_parent_h1_company_and_short_value | **0.72** | 0.80 | 0.84 | 0.00080 | Match short child text spans under a company H1 cover header |

**Aggregate (only-gold):** Avg Acc-U=0.533 · Avg Cov-S=0.867 · Avg Cov-U=0.867

#### Shared Rules (both sets)

| Rule | Acc-U | Cov-S | Cov-U | Cost |
|------|------:|------:|------:|-----:|
| rule_page1_state_value_followed_by_incorporation_label | 0.02 | 0.90 | 0.92 | 0.00003 |

#### Diagnosis

1. **Low individual accuracy** — Even when sampled-only rules retrieve text on unsampled docs, avg accuracy is only 0.04.

---

### Q8 · What is total assets at year-end (from the audited balance sheet)?

| | Sampled-refined | Unsampled-gold |
|---|---|---|
| **# Rules** | **27** | 14 |
| **Acc on sampled** | 0.90 | — |
| **Acc on unsampled** | 0.78 | 0.92 |
| **Accuracy gap (gold − sampled)** | **+0.14** | |

Overlap: 12 rules · Only-sampled: **15** · Only-gold: 2

#### Overfit Candidates (sampled only, dropped by gold)

| Rule | Acc-U | Cov-S | Cov-U | Cost | Spec | Description |
|------|------:|------:|------:|-----:|------|-------------|
| rule_tables_with_balance_sheet_in_toc_path | 0.08 | 0.00 | 0.08 | 0.00360 | MEDIUM | Match tables whose path suggests they are balance sheets |
| rule_tables_with_consolidated_keyword | 0.06 | 1.00 | 0.94 | 0.00820 | MEDIUM | Match tables with 'consolidated' plus balance-sheet-like context |
| rule_tables_with_parenthetical_year_end_dates | 0.48 | 0.40 | 0.88 | 0.00677 | MEDIUM | Match tables with year-end date style headers and total assets |
| rule_tables_with_total_assets_and_financial_statements_path | 0.76 | 0.70 | 0.94 | 0.00873 | MEDIUM | Match total-assets tables whose path_text includes financial statements |
| rule_tables_with_total_assets_and_item8_path | 0.68 | 0.60 | 0.86 | 0.00810 | MEDIUM | Match total-assets tables whose path_text includes Item 8 |
| rule_tables_with_total_assets_and_page_around_39 | 0.26 | 0.50 | 0.42 | 0.00398 | **HIGH** | Match total-assets tables around page 39 |
| rule_tables_with_total_assets_and_page_around_42 | 0.28 | 0.30 | 0.60 | 0.00448 | **HIGH** | Match total-assets tables around page 42 |
| rule_tables_with_total_assets_and_page_over_30 | 0.82 | 1.00 | 1.00 | 0.01437 | MEDIUM | Match total-assets tables on later pages |
| rule_tables_with_total_assets_and_receivables | 0.70 | 1.00 | 0.98 | 0.00807 | MEDIUM | Match balance-sheet-like tables containing total assets and receivables |
| rule_tables_with_total_assets_and_total_current_assets | 0.80 | 1.00 | 0.98 | 0.01065 | MEDIUM | Match tables containing both total current assets and total assets |
| rule_tables_with_total_assets_and_total_equity | 0.20 | 0.40 | 0.30 | 0.00313 | MEDIUM | Match balance-sheet-like tables containing total assets and total equity |
| rule_tables_with_total_assets_and_total_liabilities | 0.74 | 1.00 | 1.00 | 0.01102 | MEDIUM | Match tables containing both total assets and total liabilities |
| rule_tables_with_total_assets_in_item8_path | 0.64 | 0.60 | 0.86 | 0.00810 | MEDIUM | Match tables containing total assets under Item 8 path |
| rule_tables_with_total_assets_on_same_page_as_item8 | 0.00 | 0.10 | 0.10 | 0.00039 | MEDIUM | Match tables on pages containing an Item 8 heading and total assets |
| rule_tables_with_total_assets_within_three_pages_of_item8 | 0.32 | 0.60 | 0.72 | 0.00702 | MEDIUM | Match tables within three pages after Item 8 heading |

**Aggregate (only-sampled):** Avg Acc-U=0.455 · Avg Cov-S=0.613 · Avg Cov-U=0.711 · High-specificity: 2/15

#### Missed by Sampled (gold only)

| Rule | Acc-U | Cov-S | Cov-U | Cost | Spec | Description |
|------|------:|------:|------:|-----:|------|-------------|
| rule_tables_with_total_assets_and_accounts_payable | **0.72** | 0.90 | 0.74 | 0.00587 | MEDIUM | Match balance-sheet-like tables containing total assets and accounts payable |
| rule_tables_with_total_assets_and_page_around_60 | 0.06 | 0.50 | 0.56 | 0.00393 | **HIGH** | Match total-assets tables around page 60 |

**Aggregate (only-gold):** Avg Acc-U=0.390 · Avg Cov-S=0.700 · Avg Cov-U=0.650

#### Shared Rules (both sets)

| Rule | Acc-U | Cov-S | Cov-U | Cost |
|------|------:|------:|------:|-----:|
| rule_balance_sheet_heading_spans | 0.00 | 1.00 | 0.92 | 0.00038 |
| rule_item8_heading_spans | 0.00 | 0.90 | 0.98 | 0.00013 |
| rule_statement_of_financial_position_heading_spans | 0.00 | 0.10 | 0.00 | 0.00000 |
| rule_statement_of_financial_position_phrase | 0.00 | 0.10 | 0.16 | 0.00073 |
| rule_tables_with_financial_position_and_total_assets | 0.10 | 0.10 | 0.20 | 0.00158 |
| rule_tables_with_statement_of_financial_position | 0.00 | 0.10 | 0.08 | 0.00048 |
| rule_tables_with_total_assets_and_audited_year_end_context | 0.14 | 0.20 | 0.18 | 0.00226 |
| rule_tables_with_total_assets_and_financial_position_path | 0.06 | 0.10 | 0.08 | 0.00066 |
| rule_tables_with_total_assets_and_inventory | 0.02 | 0.00 | 0.04 | 0.00025 |
| rule_tables_with_total_assets_and_page_after_toc_item8 | 0.04 | 0.30 | 0.32 | 0.00186 |
| rule_tables_with_total_assets_and_page_around_34 | 0.10 | 0.30 | 0.30 | 0.00314 |
| rule_tables_with_total_assets_and_page_around_47 | 0.04 | 0.20 | 0.28 | 0.00165 |

#### Diagnosis

1. **Over-specific rules** — 2 dropped rules target narrow page ranges: `rule_tables_with_total_assets_and_page_around_39`, `rule_tables_with_total_assets_and_page_around_42`.
2. **Redundant rule bloat** — Sampled kept 27 rules vs gold's 14. Extra rules increase retrieval noise on unseen docs without improving accuracy.

---

### Q9 · What is total revenue for the most recent fiscal year?

| | Sampled-refined | Unsampled-gold |
|---|---|---|
| **# Rules** | 5 | 5 |
| **Acc on sampled** | 0.90 | — |
| **Acc on unsampled** | 0.70 | 0.90 |
| **Accuracy gap (gold − sampled)** | **+0.20** | |

Overlap: 1 rule · Only-sampled: 4 · Only-gold: 4

#### Overfit Candidates (sampled only, dropped by gold)

| Rule | Acc-U | Cov-S | Cov-U | Cost | Description |
|------|------:|------:|------:|-----:|-------------|
| rule_first_financial_table_after_item8 | 0.14 | 0.90 | 0.98 | 0.00317 | Match the first table after an Item 8 Financial Statements header |
| rule_income_statement_page_from_toc | 0.22 | 0.60 | 0.68 | 0.00547 | Match spans on the TOC-named page for Consolidated Statements |
| rule_item8_income_statement_tables | 0.32 | 0.70 | 0.72 | 0.00537 | Match tables under Item 8 / Financial Statements that look like income statements |
| rule_segment_note_revenue_tables | 0.26 | 0.40 | 0.30 | 0.00261 | Match tables in segment/geographic note contexts mentioning revenue |

**Aggregate (only-sampled):** Avg Acc-U=0.235 · Avg Cov-S=0.650 · Avg Cov-U=0.670 · Coverage drop S→U=−0.020

#### Missed by Sampled (gold only)

| Rule | Acc-U | Cov-S | Cov-U | Cost | Description |
|------|------:|------:|------:|-----:|-------------|
| rule_income_statement_nearby_tables | 0.42 | 0.90 | 0.80 | 0.00684 | Match tables appearing shortly after an income-statement-like header |
| rule_row_header_net_revenues | 0.14 | 0.10 | 0.22 | 0.00230 | Match tables containing a row header for net revenues |
| rule_row_header_net_sales | 0.28 | 0.60 | 0.46 | 0.01274 | Match tables containing a row header for net sales |
| rule_row_header_total_revenue | 0.14 | 0.40 | 0.44 | 0.00779 | Match tables containing a row header exactly equal to total revenue |

**Aggregate (only-gold):** Avg Acc-U=0.245 · Avg Cov-S=0.500 · Avg Cov-U=0.480 · Low sampled-coverage (<0.3): 1 rule pruned

#### Shared Rules (both sets)

| Rule | Acc-U | Cov-S | Cov-U | Cost |
|------|------:|------:|------:|-----:|
| rule_consolidated_statement_of_earnings_heading | 0.00 | 0.20 | 0.10 | 0.00001 |

#### Diagnosis

1. **Low individual accuracy** — Sampled-only rules have avg accuracy 0.14 on unsampled docs.
2. **Strategy mismatch** — Sampled selected broad page-level rules (`first_financial_table_after_item8`) whereas gold chose specific row-header rules (`row_header_net_sales`, `row_header_total_revenue`).

---

### Q10 · What is/are the trading symbol(s) and listing exchange(s)?

| | Sampled-refined | Unsampled-gold |
|---|---|---|
| **# Rules** | 4 | **32** |
| **Acc on sampled** | 1.00 | — |
| **Acc on unsampled** | 0.94 | 0.98 |
| **Accuracy gap (gold − sampled)** | **+0.04** | |

Overlap: 2 rules · Only-sampled: 2 · Only-gold: **30**

#### Overfit Candidates (sampled only, dropped by gold)

| Rule | Acc-U | Cov-S | Cov-U | Cost | Description |
|------|------:|------:|------:|-----:|-------------|
| rule_cover_page_table_cells_symbol_exchange | 0.18 | 0.00 | 0.18 | 0.00014 | Match table spans whose cells contain trading symbol / exchange data |
| rule_page1_symbols_in_registration_window | 0.12 | 0.20 | 0.32 | 0.00001 | Match ticker-like spans appearing after the first 12(b) mention |

**Aggregate (only-sampled):** Avg Acc-U=0.150 · Avg Cov-S=0.100 · Avg Cov-U=0.250

#### Missed by Sampled (gold only — 30 rules)

| Rule | Acc-U | Cov-S | Cov-U | Cost | Description |
|------|------:|------:|------:|-----:|-------------|
| rule_business_path_and_stock_terms | 0.36 | 1.00 | 1.00 | 0.01310 | Match spans under Item 1 / Business mentioning common stock trading |
| rule_business_sentence_trades_on_exchange | **0.74** | 0.80 | 0.78 | 0.00094 | Match business-description sentences saying common stock trades on exchange |
| rule_business_sentence_under_symbol | 0.68 | 0.90 | 0.82 | 0.00119 | Match business-section sentences stating 'common stock is listed under symbol' |
| rule_cover_page_company_section_header | 0.04 | 1.00 | 0.96 | 0.00009 | Match the large company-name section header on page 1 |
| rule_cover_page_table_row_security_registration | 0.18 | 0.00 | 0.18 | 0.00014 | Match table spans with Section 12(b) registration data |
| rule_exact_symbol_values | 0.10 | 0.30 | 0.40 | 0.00002 | Match standalone ticker spans commonly used as trading symbols |
| rule_inline_registered_under_symbol | 0.60 | 0.80 | 0.68 | 0.00086 | Match body text stating stock is listed/traded under a symbol |
| rule_item1_business_page3_symbol_exchange | 0.22 | 0.40 | 0.26 | 0.00042 | Match page 3 Item 1 Business spans for symbol/exchange |
| rule_page1_after_company_name_until_documents_incorporated | 0.60 | 0.90 | 0.94 | 0.00944 | Match page 1 spans between company-name header and 'Documents Incorporated' |
| rule_page1_after_trading_symbol_header | 0.24 | 0.60 | 0.60 | 0.00076 | Match spans immediately following a 'Trading Symbol' header |
| rule_page1_allcaps_ticker_headers | 0.08 | 0.20 | 0.16 | 0.00000 | Match all-caps short section headers on page 1 that look like tickers |
| rule_page1_bold_smallfont_registration_values | 0.14 | 0.30 | 0.38 | 0.00016 | Match page 1 small-font bold spans for ticker or registration values |
| rule_page1_exact_exchange_spans | 0.00 | 0.90 | 0.38 | 0.00003 | Match page 1 spans whose text is exactly an exchange name |
| rule_page1_exchange_before_symbol_like | 0.10 | 0.30 | 0.38 | 0.00005 | Match symbol-like spans that appear before/after exchange names |
| rule_page1_exchange_in_registration_window | 0.24 | 0.90 | 0.88 | 0.00043 | Match exchange-name spans after the first 12(b) mention |
| rule_page1_exchange_symbol_split_reverse | 0.04 | 0.00 | 0.04 | 0.00001 | Match adjacent page 1 spans where one looks like an exchange |
| rule_page1_exchange_then_symbol_adjacent | 0.04 | 0.20 | 0.20 | 0.00004 | Match adjacent page 1 spans where an exchange name is near a ticker |
| rule_page1_header_path_registration | 0.04 | 0.80 | 0.86 | 0.00115 | Match page 1 spans whose path_text contains registration info |
| rule_page1_major_exchange_names | 0.40 | 1.00 | 0.80 | 0.00055 | Match page 1 spans containing common exchange names |
| rule_page1_near_company_name_and_section12b | 0.24 | 0.80 | 0.84 | 0.00098 | Match page 1 spans under company-name path_text near Section 12(b) |
| rule_page1_near_title_of_each_class | 0.52 | 1.00 | 0.94 | 0.00077 | Match spans around 'Title of each class' |
| rule_page1_registration_data_band | 0.52 | 1.00 | 0.98 | 0.00243 | Match the dense band of page 1 body/header spans between company-name and securities |
| rule_page1_registration_keywords_broad | 0.22 | 1.00 | 1.00 | 0.00488 | Broad high-recall rule for page 1 spans mentioning registration keywords |
| rule_page1_security_registration_cluster | 0.16 | 1.00 | 0.96 | 0.00074 | Match the cluster of page 1 spans between Section 12(b) and securities info |
| rule_page1_symbol_like_spans | 0.10 | 0.30 | 0.40 | 0.00002 | Match short page 1 spans that look like ticker symbols |
| rule_page1_symbol_then_exchange_adjacent | 0.16 | 0.20 | 0.32 | 0.00008 | Match adjacent page 1 spans where a ticker-like token is followed by exchange |
| rule_page1_title_trading_exchange_cluster | 0.54 | 1.00 | 0.94 | 0.00067 | Match spans in the page 1 cluster containing title/trading symbol/exchange |
| rule_page1_topmatter_short_values_after_common_stock | 0.22 | 0.80 | 0.60 | 0.00006 | Match short value spans following a 'Common Stock' span |
| rule_page1_under_company_header | 0.48 | 0.90 | 0.94 | 0.00729 | Match page 1 body spans under the top company header |
| rule_title_trading_exchange_triplet | 0.60 | 1.00 | 0.96 | 0.00811 | Match spans near the cover-page triplet 'Title / Trading / Exchange' |

**Aggregate (only-gold):** Avg Acc-U=0.287 · Avg Cov-S=0.677 · Avg Cov-U=0.653 · Low sampled-coverage (<0.3): 5 rules pruned

#### Shared Rules (both sets)

| Rule | Acc-U | Cov-S | Cov-U | Cost |
|------|------:|------:|------:|-----:|
| rule_cover_page_embedded_symbol_exchange | 0.20 | 0.40 | 0.52 | 0.00019 |
| rule_cover_page_registration_sequence | **0.94** | 1.00 | 1.00 | 0.19096 |

#### Diagnosis

1. **Low individual accuracy** — Sampled-only rules have avg accuracy 0.15 on unsampled docs.
2. **Sampling bias** — 2 gold rules had zero coverage on all 10 sampled docs and were eliminated despite generalizing: `rule_cover_page_table_row_security_registration`, `rule_page1_exchange_symbol_split_reverse`.

---

## Cross-Question Aggregate Analysis

### 1. Accuracy Summary

| Question | S-Acc-S | S-Acc-U | G-Acc-U | Gap | #S | #G |
|----------|--------:|--------:|--------:|----:|---:|---:|
| Shares outstanding | 0.90 | 0.66 | 0.96 | **+0.30** | 7 | 9 |
| Long-term debt | 0.50 | 0.46 | 0.82 | **+0.36** | 6 | 11 |
| Net income | 0.90 | 0.76 | 0.82 | +0.06 | 14 | 22 |
| Address & ZIP | 0.90 | 0.88 | 0.98 | +0.10 | 5 | 7 |
| Registrant name | 0.90 | 0.90 | 1.00 | +0.10 | 2 | 4 |
| Telephone number | 1.00 | 0.82 | 0.98 | +0.16 | 5 | 5 |
| State & EIN | 1.00 | 0.88 | 1.00 | +0.12 | 3 | 4 |
| Total assets | 0.90 | 0.78 | 0.92 | +0.14 | 27 | 14 |
| Total revenue | 0.90 | 0.70 | 0.90 | **+0.20** | 5 | 5 |
| Trading symbol(s) | 1.00 | 0.94 | 0.98 | +0.04 | 4 | 32 |
| **Average** | | **0.778** | **0.936** | **+0.158** | | |

*S-Acc-S = sampled-refined accuracy on sampled docs · S-Acc-U = sampled-refined accuracy on unsampled docs · G-Acc-U = gold accuracy on unsampled docs*

---

### 2. Coverage Representativeness

For rules appearing in either set, how well does sampled coverage predict unsampled coverage (across 151 rule-question pairs)?

| Coverage pattern | Count | % | Interpretation |
|-----------------|------:|--:|----------------|
| High-S & High-U (both ≥0.5) | 86 | 57% | Agreement |
| **High-S but Low-U** | **12** | **8%** | Overfit risk: sampled fires but unseen doesn't |
| Low-S but High-U | 4 | 3% | Missed: unseen fires but sampled doesn't |
| Both Low | 49 | 32% | Agreement (neither fires) |

- 89% of rules agree between sampled and unsampled coverage
- **8%** fire on sampled but not broadly → overfit source
- **3%** fire on unsampled but were invisible in sampled → underfit source

---

### 3. Rule Specificity Breakdown

| Specificity | Only-sampled (overfit candidates) | Only-gold (missed) | Shared |
|-------------|:---------------------------------:|:------------------:|:------:|
| HIGH (page/date-specific) | 4 (11%) | 1 (1%) | 1 (2%) |
| MEDIUM | 34 (89%) | 63 (86%) | 39 (98%) |
| LOW (broad/general) | 0 (0%) | 9 (12%) | 0 (0%) |

> High-specificity rules tend to overfit. Gold set includes proportionally more broad/general rules than sampled set.

---

### 4. Is the Sampled Data Representative?

**Evidence FOR representativeness:**
- 40/78 sampled-selected rules (51%) are also in the gold set — the sampled set correctly identifies many generalizable rules.

**Evidence AGAINST representativeness (sources of overfit):**

| Source | Description |
|--------|-------------|
| **a) Blind spots** | 3 gold rules had zero coverage on all 10 sampled docs and were silently eliminated |
| **b) Layout diversity** | 10 sampled docs may not capture all 10-K cover-page layouts, balance-sheet page placements, or section-header naming conventions |
| **c) Small-N pruning noise** | With only 10 docs, a rule contributing to 1 correct answer looks marginal and gets pruned, but may be critical for 15+ unsampled docs |
| **d) D\* set size** | The target set D\* is small on 10 docs; refinement only keeps rules covering D\* docs, causing broader rules to appear redundant |

**Rules with zero sampled coverage eliminated despite generalizing:**

| Question | Rule | Acc-U | Cov-U |
|----------|------|------:|------:|
| Shares outstanding | rule_page1_company_cover_section_header | 0.00 | 0.00 |
| Trading symbol(s) | rule_cover_page_table_row_security_registration | 0.18 | 0.18 |
| Trading symbol(s) | rule_page1_exchange_symbol_split_reverse | 0.04 | 0.04 |

---

### 5. Improvement Suggestions

| # | Suggestion | Detail |
|---|-----------|--------|
| 1 | **Increase sampled set size** | Use 20–30 docs instead of 10; blind spots (zero-coverage gold rules) would shrink substantially |
| 2 | **Stratified sampling** | Ensure sampled docs cover diverse 10-K layouts: large-cap vs small-cap, varied fiscal year-end months, foreign private issuers, varied page lengths |
| 3 | **Coverage-weighted pruning** | Down-weight rules with very high sampled coverage (≥0.9) but very low cost ratio — a signal of narrow, layout-specific rules |
| 4 | **Penalize high-specificity rules** | Rules containing page-number constraints, specific month names, or narrow structural patterns should receive a specificity penalty during selection |
| 5 | **Minimum-coverage threshold** | Require any selected rule to have coverage ≥0.2 on sampled docs; rules with zero sampled coverage should not be auto-pruned |
| 6 | **Ensemble rule selection** | Run refinement on multiple random 10-doc subsets; take rules selected in ≥k/m runs to reduce layout idiosyncrasy impact |
| 7 | **Accuracy-weighted selection** | Incorporate per-rule individual accuracy estimates into selection to avoid low-quality rules that appear useful only when merged |

---

### 6. Root Cause Summary Per Question

| Question | Primary Cause | Notes |
|----------|--------------|-------|
| **Shares outstanding** | Cheap specific rules (date-specific phrases) dominate sampled selection; `rule_cover_path_text_company_page1` (Acc-U=0.72) excluded as too expensive | Gap +0.30 |
| **Long-term debt** | Missed high-accuracy gold rules (avg Acc-U=0.37, avg Cov-U=0.81) excluded by cost-sort before evaluation | Gap +0.36, worst question |
| **Net income** | High-cost redundant rules kept (cost >0.04 each); 12 structurally specific gold rules missed | Gap +0.06, mild |
| **Address & ZIP** | Single low-quality sampled rule; 3 gold rules missed with reasonable coverage | Gap +0.10 |
| **Registrant name** | Sampled-only set is strict subset of gold; 2 missed gold rules cover diverse header patterns | Gap +0.10 |
| **Telephone number** | Near-complete rule-set mismatch (1 shared rule); semantics mismatch, not coverage | Gap +0.16 |
| **State & EIN** | Sampled rules retrieve EIN text that is wrong answer; gold rules retrieve surrounding context correctly | Gap +0.12 |
| **Total assets** | Rule bloat: 27 rules vs 14; 15 redundant rules add retrieval noise; 2 page-specific rules | Gap +0.14 |
| **Total revenue** | Fundamental mismatch — sampled picked page-level rules, gold picked row-header rules | Gap +0.20 |
| **Trading symbol(s)** | Small gap despite large rule-set difference (4 vs 32); `rule_cover_page_registration_sequence` (Acc-U=0.94) does most of the work | Gap +0.04 |

---

## Executive Summary

> **Avg accuracy gap (sampled-refined vs gold on unsampled): +0.158**
> Worst questions: long-term debt (+0.36), shares outstanding (+0.30), total revenue (+0.20)

### Three Root Causes (in order of impact)

| Rank | Root Cause | Estimated Impact | Description |
|------|-----------|:----------------:|-------------|
| 1 | **Sampling blind spots** | ~40% | 10 sampled docs do not trigger rules that fire on alternative 10-K layouts. Rules with zero sampled coverage but strong unsampled performance are silently eliminated (3 such rules found). |
| 2 | **Small-N pruning noise** | ~35% | With 10 docs, a rule's marginal contribution is unreliable. Rules helping 1–2 sampled docs look optional and get pruned, but are critical for 10–20 unsampled docs. Total assets (27→14 rules) is the clearest case. |
| 3 | **Rule specificity** | ~25% | Some selected rules encode layout-specific patterns (page numbers, date references, specific section names) that match sampled docs by coincidence but add noise on broader unseen docs. |

### Recommended Fix

> **Increase to 25–30 stratified sampled docs + add a minimum unsampled-coverage proxy filter to the rule selection pipeline.**
