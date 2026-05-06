# Refined Rules Summary — Shares Outstanding Question

**Question:** How many shares of common stock were outstanding as of the cover-page reference date?  
**Selected rules:** 7 / 100  
**Sampled acc:** 0.90 → **Unsampled acc:** 0.66 (significant drop — overfitting to sample)

---

## Rules and What They Encode

### 1. `rule_page1_phrase_number_of_shares_outstanding_of_registrant`
**Pattern:** Page 1 spans containing both `"number of shares outstanding"` and `"registrant"`  
**Encodes:** The SEC cover page boilerplate phrase. Most 10-Ks use this exact phrasing in the cover page header row.  
**Weakness:** Misses filings that omit "registrant" or use slightly different phrasing.

---

### 2. `rule_page1_sentence_with_there_were_and_common_stock`
**Pattern:** Page 1 spans containing `"there were"` + `"common stock"` + `"outstanding"`  
**Encodes:** Narrative-style phrasing used in some cover pages: *"As of [date], there were X shares of common stock outstanding."*  
**Weakness:** Only fires for the "there were" formulation — misses "as of [date], [X] shares were outstanding" and other variants.

---

### 3. `rule_page1_shares_outstanding_as_of_february`
**Pattern:** Pages 1–2, contains `"outstanding"` and matches regex `"as of february \d{1,2}, \d{4}"`  
**Encodes:** February as the cover-page reference date — common for calendar-year filers whose fiscal year ends December 31 (cover page filed in February).  
**Weakness:** Hardcodes February. Misses companies with non-calendar fiscal years (e.g., fiscal year ending in June → August reference date) or early filers (January).

---

### 4. `rule_page1_numeric_after_phrase_as_of_january`
**Pattern:** Page 1 standalone numeric spans (≥6 digits) preceded within 6 spans by `"as of january"` + (`"outstanding"` or `"number of shares"`)  
**Encodes:** January as the cover-page reference date — for companies filing very early or with non-standard fiscal year ends.  
**Weakness:** Only covers January; misses all other months. The 6-span lookback window may be too narrow for some layouts.

---

### 5. `rule_h1_company_cover_span_with_outstanding`
**Pattern:** Page 1, label=`section_header`, level=`H1`, text contains `"outstanding"`  
**Encodes:** Some filings embed the shares-outstanding sentence directly into the large H1 company-name header block on the cover page.  
**Weakness:** Rare pattern — most H1 headers are just the company name. Likely only fired for 1–2 docs in the sample.

---

### 6. `rule_page1_cover_text_with_multiple_large_numbers`
**Pattern:** Pages 1–2, label=`text` or `section_header`, contains ≥2 numbers each with ≥6 digits  
**Encodes:** Cover page blocks that contain multiple large numbers (e.g., shares + par value + EIN or zip code). A broad catch-all for structured cover blocks.  
**Weakness:** Very broad — retrieves large chunks of text when multiple unrelated numbers appear together. High recall, low precision.

---

### 7. `rule_page2_cover_page_numeric_sentence`
**Pattern:** Page 2, contains a long comma-formatted number (≥3 comma groups) + `"outstanding"` + (`"common stock"` or `"shares"`)  
**Encodes:** Spillover cover pages — some filings push the shares-outstanding sentence to page 2 when the cover page is long.  
**Weakness:** Only checks page 2. Misses spillovers to page 3+. The 3-comma-group requirement may filter out smaller share counts.

---

## Why Accuracy Drops on Unsampled Docs (0.90 → 0.66)

| Root Cause | Rules Affected |
|---|---|
| **Month hardcoding** — only January and February covered | rules 3, 4 |
| **Narrow span lookback** — misses split layouts | rule 4 |
| **Broad catch-all fires on wrong spans** | rule 6 |
| **Page 2 only for spillover** | rule 7 |
| **H1 pattern very rare** | rule 5 |

The 100 original rules covered many more month patterns (March–December) and layout variations. Rule refinement pruned those as redundant on the 10 sampled docs (which happened to all be February/January filers or standard layouts), leaving gaps for the remaining 32 unsampled docs.
