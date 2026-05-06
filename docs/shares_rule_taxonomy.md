# Rule Taxonomy — Shares Outstanding Question

**Question:** How many shares of common stock were outstanding as of the cover-page reference date?  
**Total rules:** 100

---

## Taxonomy

### Type 1 — Exact Phrase Match (15 rules)
Match a span containing a specific, complete phrase that directly names the answer field. High precision, brittle to paraphrasing.

| Rule | Phrase |
|---|---|
| `rule_page1_phrase_common_shares_outstanding` | "common shares outstanding" |
| `rule_page1_phrase_common_stock_issued_and_outstanding_as_of` | "common stock issued and outstanding as of" |
| `rule_page1_phrase_common_stock_outstanding_as_of` | "common stock outstanding as of" |
| `rule_page1_phrase_number_of_shares_of_common_stock_outstanding` | "number of shares of common stock outstanding" |
| `rule_page1_phrase_number_of_shares_outstanding_of_registrant` | "number of shares outstanding of the registrant" |
| `rule_page1_phrase_shares_of_common_stock_issued_and_outstanding` | "shares of common stock issued and outstanding" |
| `rule_page1_phrase_shares_of_common_stock_outstanding_as_of` | "shares of common stock outstanding as of" |
| `rule_page1_common_shares_outstanding` | "common shares outstanding" |
| `rule_page1_common_stock_issued_and_outstanding` | "shares of common stock issued and outstanding" |
| `rule_page1_common_stock_outstanding_as_of` | "common stock ... outstanding as of" |
| `rule_page1_issued_and_outstanding_sentence` | "issued and outstanding" |
| `rule_page1_shares_outstanding_issued_and_outstanding` | "shares of common stock issued and outstanding as of" |
| `rule_page1_registrant_common_stock_outstanding` | "registrant's common stock outstanding" |
| `rule_page1_registrants_common_stock_outstanding` | "registrant's common stock outstanding" |
| `rule_page2_issued_and_outstanding_sentence` | "issued and outstanding" (page 2) |

---

### Type 2 — Keyword Co-occurrence (22 rules)
Match a span containing two or more keywords that together signal the answer field. Less brittle than exact phrases; handles paraphrasing.

| Rule | Keywords Required |
|---|---|
| `rule_page1_outstanding_sentence` | "outstanding" + ("common stock" or "shares") |
| `rule_page1_outstanding_sentence_with_common` | "common" + "outstanding" |
| `rule_page1_outstanding_sentence_with_shares` | "shares" + "outstanding" |
| `rule_page1_outstanding_sentence_with_registrant` | "registrant" + "outstanding" |
| `rule_page1_sentence_with_registrant_and_outstanding` | "registrant" + "outstanding" |
| `rule_page1_shares_of_common_stock_outstanding` | "shares of common stock outstanding" |
| `rule_page1_number_of_shares_outstanding` | "number of shares" + "outstanding" |
| `rule_page1_common_stock_outstanding_text_only` | "common stock" + "outstanding" (text label only) |
| `rule_page1_shares_outstanding_text_only` | "shares outstanding" (text label only) |
| `rule_page1_shares_outstanding_as_of` | "outstanding" + "as of" |
| `rule_page1_as_of_date_outstanding` | "as of" + "outstanding" |
| `rule_page1_cover_span_with_as_of_and_common_stock` | "as of" + "common stock" |
| `rule_page1_cover_span_with_as_of_and_issued_outstanding` | "as of" + "issued and outstanding" |
| `rule_page1_cover_span_with_as_of_and_number_of_shares` | "as of" + "number of shares of common stock outstanding" |
| `rule_page1_cover_page_stock_sentence` | "as of" + "outstanding" |
| `rule_page1_sentence_with_common_stock_and_date` | "common stock" + date pattern |
| `rule_page1_sentence_with_shares_and_date` | "shares" + date pattern |
| `rule_page1_text_span_with_outstanding_and_date` | "outstanding" + month-day-year date |
| `rule_page1_text_span_with_shares_and_date` | "shares" + month-day-year date |
| `rule_page1_sentence_with_number_of_shares_of_common_stock` | "number of shares of common stock outstanding" |
| `rule_page1_sentence_with_number_of_shares_of_registrants_common_stock` | "number of shares of the registrant's common stock" |
| `rule_page1_sentence_with_shares_of_common_stock_issued_and_outstanding` | "shares of common stock issued and outstanding" |

---

### Type 3 — Verb / Sentence Pattern (5 rules)
Match a span by the grammatical structure of the sentence, not just keywords.

| Rule | Pattern |
|---|---|
| `rule_page1_sentence_with_there_were_and_common_stock` | "there were" + "common stock" + "outstanding" |
| `rule_page1_there_were_common_stock_outstanding` | begins with "there were ... shares of common stock outstanding" |
| `rule_page1_there_were_shares_outstanding` | begins with "there were ... shares ... outstanding" |
| `rule_page1_sentence_with_common_stock_outstanding_and_documents_reference` | "common stock outstanding" within a block also containing "documents incorporated" |
| `rule_page1_market_value_and_outstanding_same_span` | market value language + outstanding language in same span |

---

### Type 4 — Date-anchored (8 rules)
Anchor retrieval on the specific month mentioned in the "as of" reference date. High precision for companies with known filing patterns; completely miss other months.

| Rule | Month(s) |
|---|---|
| `rule_page1_shares_outstanding_as_of_january` | January |
| `rule_page1_shares_outstanding_as_of_february` | February |
| `rule_page1_shares_outstanding_as_of_october` | October |
| `rule_page1_numeric_after_phrase_as_of_january` | January (numeric span after label) |
| `rule_page1_numeric_after_phrase_as_of_february` | February (numeric span after label) |
| `rule_page1_numeric_after_phrase_as_of_october` | October (numeric span after label) |
| `rule_page1_outstanding_as_of_january_or_february` | January or February |
| `rule_page1_numeric_near_january_february_date` | January or February (numeric near date) |

---

### Type 5 — Positional Context (anchor landmark, 14 rules)
Retrieve spans at a specific position relative to a structural landmark in the cover page (e.g., "after market value", "before documents incorporated"). Requires the landmark to be present and correctly parsed.

| Rule | Landmark | Direction |
|---|---|---|
| `rule_page1_after_market_value_sentence` | market value sentence | after |
| `rule_page1_after_market_value_before_documents` | market value / documents incorporated | between |
| `rule_page1_after_nonaffiliate_market_value` | non-affiliate market value sentence | after |
| `rule_page1_after_shell_company_question` | shell company checkbox | after |
| `rule_page1_before_documents_incorporated` | "DOCUMENTS INCORPORATED BY REFERENCE" | before |
| `rule_page1_outstanding_near_documents_incorporated` | "DOCUMENTS INCORPORATED BY REFERENCE" | near/before |
| `rule_page1_near_documents_incorporated_preceding` | "DOCUMENTS INCORPORATED BY REFERENCE" | immediately before |
| `rule_page1_cover_span_with_outstanding_before_documents_reference` | "DOCUMENTS INCORPORATED BY REFERENCE" | before |
| `rule_page1_cover_block_with_documents_incorporated` | "DOCUMENTS INCORPORATED BY REFERENCE" | near |
| `rule_page1_cover_block_before_part_i` | "Part I" | before |
| `rule_page1_sentence_with_market_value_and_outstanding_same_block` | market value disclosure | same block |
| `rule_page1_market_value_and_number_of_shares_labels` | market value label | adjacent |
| `rule_page1_near_company_name_cover_block` | company name cover block | under/near |
| `rule_page1_cover_page_top_half_outstanding` | top half of page 1 | positional cutoff |

---

### Type 6 — Split Label-Number Layout (10 rules)
Handle filings where the label (e.g., "Number of shares of common stock outstanding as of ...") and the answer (a bare number like "543,210,000") appear in separate, adjacent spans. The rule captures the numeric span by finding it near the label span.

| Rule | Strategy |
|---|---|
| `rule_page1_numeric_only_after_outstanding_label` | numeric span following an outstanding-label span |
| `rule_page1_numeric_only_near_number_of_shares_label` | numeric span near "number of shares outstanding as of" label |
| `rule_page1_numeric_span_after_common_stock_label` | numeric span within 5 spans of "common stock outstanding" label |
| `rule_page1_numeric_span_after_number_of_shares_label` | numeric span immediately after "number of shares" label |
| `rule_page1_or_2_cover_page_numeric_only_after_label` | standalone numeric span near outstanding label (pages 1–2) |
| `rule_page1_standalone_large_number_after_outstanding_label` | standalone numeric after "Number of shares ... outstanding as of ..." |
| `rule_page1_standalone_number_after_market_value_and_outstanding_label` | standalone numeric in two-column layout with market value |
| `rule_page1_two_column_label_number_pattern` | prior span is an outstanding label in two-column layout |
| `rule_page1_numeric_near_as_of_date_and_outstanding` | numeric near "as of" + "outstanding" |
| `rule_page1_numeric_span_after_common_stock_label` | numeric span following common stock label |

---

### Type 7 — Document Structure / Semantic Block (9 rules)
Target specific structural elements: heading levels (H1/H2/H3), body depth, or the section_header label. Relies on the document parser correctly labeling elements.

| Rule | Structural Signal |
|---|---|
| `rule_h1_company_cover_span_with_outstanding` | label=section_header, level=H1, "outstanding" in text |
| `rule_page1_cover_h1_company_block_with_outstanding` | section_header, H1, outstanding inline |
| `rule_page1_cover_h2_block_with_outstanding` | H2 or H3 cover block with outstanding inline |
| `rule_page1_company_cover_section_header` | main company cover section_header |
| `rule_page1_cover_page_body_depth2_or_3_outstanding` | body depth 2 or 3 on page 1 |
| `rule_page1_cover_page_company_block_text_span_outstanding` | section_header whose text_span contains outstanding sentence |
| `rule_page1_cover_page_small_font_outstanding` | small font attribute + outstanding language |
| `rule_page1_cover_page_reference_date_labels` | label spans defining cover-page reference date |
| `rule_page1_market_value_and_number_of_shares_labels` | side-by-side label structure (market value + shares) |

---

### Type 8 — Hierarchical Path (4 rules)
Use the document's structural path metadata (the hierarchy of section names leading to a span) to locate spans under the company cover block rather than under Item sections.

| Rule | Path Signal |
|---|---|
| `rule_cover_path_text_company_page1` | path_text exists and is not "Form 10-K" + outstanding/common stock in text |
| `rule_page1_any_common_stock_in_company_path` | non-Item path + "common stock" |
| `rule_page1_any_outstanding_in_company_path` | non-Item path + "outstanding" |
| `rule_page1_path_company_name_and_outstanding` | company cover-page path + "outstanding" |

---

### Type 9 — Special Layout: Dual-Class Stock (4 rules)
Handle filings with two share classes (Class A + Class B) where the answer is the total, appearing after per-class disclosures. Relies on detecting "Class A" and "Class B" in nearby spans.

| Rule | Strategy |
|---|---|
| `rule_page1_total_shares_after_class_a_class_b` | numeric span after Class A and Class B counts |
| `rule_page1_total_shares_after_class_a_class_b_without_outstanding` | same, even if "outstanding" is only in nearby label |
| `rule_page1_two_class_total_outstanding_number` | standalone total in dual-class layout |
| `rule_page1_combined_market_value_and_share_count_numbers` | span with two large numbers (market value + share count) |

---

### Type 10 — Broad Catch-all / High-Recall (9 rules)
Return large sets of spans with minimal filtering. Used as fallback coverage for unusual layouts. Low precision; intended to combine with Type 1–9 rules in a merge strategy.

| Rule | Scope |
|---|---|
| `rule_page1_page2_only` | ALL page 1–2 spans with "outstanding" or "common stock" or "shares" |
| `rule_page1_or_2_cover_page_outstanding` | pages 1–2 with cover-page outstanding language |
| `rule_page1_cover_numeric_candidates` | all large numeric spans on pages 1–2 |
| `rule_page1_cover_numeric_candidates_excluding_dollars` | large numeric spans excluding dollar amounts |
| `rule_page1_cover_numeric_spans_large` | large comma-formatted numeric spans on pages 1–2 |
| `rule_page1_numeric_in_company_cover_block` | large numeric spans under company-name cover paths |
| `rule_page1_cover_text_with_multiple_large_numbers` | pages 1–2 spans with ≥2 large numbers |
| `rule_page1_text_span_contains_outstanding` | any page-1 span with "outstanding" in text or text_span |
| `rule_page2_text_span_contains_outstanding` | any page-2 span with "outstanding" in text |

---

## Summary Table

| Type | Description | Count | Precision | Recall |
|---|---|---|---|---|
| 1 — Exact Phrase | Full phrase match | 15 | High | Low |
| 2 — Keyword Co-occurrence | 2+ keywords in same span | 22 | Medium | Medium |
| 3 — Verb / Sentence Pattern | Grammatical structure | 5 | High | Low |
| 4 — Date-anchored | Month in "as of" date | 8 | High | Very Low (month-specific) |
| 5 — Positional Context | Relative to landmark | 14 | Medium | Medium |
| 6 — Split Label-Number | Numeric span near label | 10 | High | Medium |
| 7 — Document Structure | H1/H2/depth/font | 9 | Medium | Medium |
| 8 — Hierarchical Path | path_text metadata | 4 | Medium | Low |
| 9 — Dual-Class Layout | Class A + Class B total | 4 | High | Very Low (layout-specific) |
| 10 — Broad Catch-all | Minimal filter | 9 | Low | High |
| **Total** | | **100** | | |

---

## Key Observations

1. **Phrase redundancy**: Types 1–3 overlap heavily. ~20 rules encode nearly identical phrases with minor wording variations ("common shares outstanding" vs "shares of common stock outstanding"). The LLM generates these because it cannot test which exact phrasing each company uses.

2. **Date hardcoding is a major coverage gap**: Type 4 covers only January, February, and October. Non-calendar-year filers (March, April, May, June, July, August, September, November, December) are only caught by Types 1–3 or 10. This is the primary cause of the 0.66 unsampled accuracy.

3. **Split layout rules are essential**: Many 10-Ks use a two-column cover page where the label and number are separate spans. Type 6 rules handle this; without them, the extracted text contains only the label with no number, and the QA LLM answers "NOT FOUND".

4. **Dual-class rules are rare but necessary**: A small subset of companies (e.g., Alphabet, Meta) have Class A + Class B shares. Without Type 9 rules, these answers are missed entirely.

5. **Broad catch-alls ensure coverage but inflate cost**: Type 10 rules retrieve entire page 1–2 contents. They are kept as fallback but significantly increase the retrieved token count (cost).
