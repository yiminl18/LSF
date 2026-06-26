# Datasets

This document summarizes the four datasets used in the LSF project. Each dataset
follows the same structure under `data/<dataset>/`:

- `all_labels.json` — maps each document → question → ground-truth answer
- `queries.json` / `multi_cluster_queries.txt` — the list of extraction questions (with answer types)
- `text/`, `json/` (and `processing/*_reconstructed.json`) — raw text and structured document representations

> **Note on links:** The repository does not record source URLs for any dataset
> (document JSONs carry only `filename`/`mimetype`/`binary_hash`). The "Source /
> link" entries below are the canonical public portals each corpus was drawn
> from, provided for reference.

| Dataset | Domain | Documents | Questions |
|---|---|---|---|
| [FinanceBench](#1-financebench) | Financial / SEC filings | 100 | 12 |
| [Court](#2-court) | Legal / federal appeals | 294 | 13 |
| [NoPV](#3-nopv) | Regulatory / pipeline safety | 242 | 12 |
| [OfficeQA](#4-officeqa) | Economic / U.S. Treasury | 200 | 16 |
| [Publications](#5-publications) | Academic / scientific papers | — | — |
| [Medical Records](#6-medical-records) | Clinical / patient records | — | — |
| [NHC Tropical Cyclone Reports](#7-nhc-tropical-cyclone-reports-nhc_tcr) | Meteorology / storm reports | — | 14 |
| [EMA EPAR Product Information](#8-ema-epar-product-information-epar) | Pharmaceutical / drug labels | — | 13 |

---

## 1. FinanceBench

**Description:** SEC financial filings (10-K and 10-Q) from major publicly traded
companies, used for extracting company registration details, financial metrics,
and disclosure items.

**Source / link:** SEC EDGAR — https://www.sec.gov/edgar (filings; the
FinanceBench benchmark by Patronus AI: https://github.com/patronus-ai/financebench)

**Common extractable values:**

1. Registrant's exact name
2. State/jurisdiction of incorporation and IRS Employer ID Number (EIN)
3. Principal executive offices address (city and state/country) and ZIP code
4. Registrant's telephone number
5. Reporting period / fiscal year-end date
6. Document type / SEC form (e.g., 10-K, 10-Q)
7. Stock exchange the common stock is listed on, with trading symbol(s)
8. Number of outstanding shares of common stock
9. Reportable operating segments
10. Principal products/services and principal markets/geographies
11. Total revenue and net income (loss) for the most recent fiscal year
12. Total assets and long-term debt at year-end
13. Audit opinion type and independent accounting firm name

**Source files:** `data/financebench/all_labels.json`,
`data/financebench/multi_cluster_queries.txt`,
`data/financebench/processing/*_reconstructed.json`

---

## 2. Court

**Description:** U.S. federal court appeal opinions, used for extracting case
details, judges, procedural information, and case dispositions.

**Source / link:** CourtListener — https://www.courtlistener.com (U.S. federal
courts: https://www.uscourts.gov)

**Common extractable values:**

1. Court of Appeals docket number(s)
2. Originating district court docket number(s) ("D.C. No.")
3. Originating U.S. District Court(s)
4. District judge(s) who presided below
5. Date the case was argued or submitted on briefs
6. City where the case was argued/submitted
7. Opinion filing date
8. Panel judges that decided the case
9. Judge who authored the majority opinion
10. Legal subject matter / topic
11. First-listed attorney for the appellant(s)
12. First-listed attorney for the appellee(s)
13. Panel's final disposition (AFFIRMED / REVERSED / REMANDED, etc.)

**Source files:** `data/court/all_labels.json`, `data/court/queries.json`,
`data/court/text/`, `data/court/json/`

---

## 3. NoPV

**Description:** PHMSA (Pipeline and Hazardous Materials Safety Administration)
Notices of Probable Violation, used for extracting operator details, case
information, regulatory citations, and corrective-action requirements.

**Source / link:** PHMSA Pipeline Safety enforcement —
https://primis.phmsa.dot.gov/comm/reports/enforce/

**Common extractable values:**

1. Legal name of the pipeline operator
2. CPF case number (Compliance Program File)
3. PHMSA Region office that issued the notice (Eastern/Southern/Central/Western/Southwest)
4. Date the Notice of Probable Violation was issued
5. Inspection date or date range
6. U.S. state(s) where the pipeline facility is located
7. 49 CFR Part covering the regulated activity (Part 192 / 195 / 199)
8. Number of distinct alleged violations
9. All 49 CFR section numbers cited for violations
10. Number of distinct corrective-action items
11. Longest deadline (in days) for completing corrective action
12. Name of the person who signed the notice (Regional Director / Director)

**Source files:** `data/nopv/all_labels.json`, `data/nopv/queries.json`,
`data/nopv/text/`, `data/nopv/json/`

---

## 4. OfficeQA

**Description:** U.S. Treasury Bulletins (periodic economic and financial
reports), used for extracting macroeconomic indicators, federal budget
information, debt metrics, and currency data.

**Source / link:** U.S. Treasury Bulletin —
https://fiscal.treasury.gov/reports-statements/treasury-bulletin/

**Common extractable values:**

1. Quarter and year covered by the bulletin
2. Annualized real GDP growth rate (most recent quarter)
3. Unemployment rate (most recent month)
4. Average monthly nonfarm payroll job growth (year-to-date)
5. Consumer sentiment index (University of Michigan / Reuters)
6. Federal budget deficit as a percentage of GDP
7. Total federal receipts (fiscal year to date)
8. Total surplus or deficit (fiscal year to date)
9. Net individual income tax receipts (most recent quarter)
10. Total gross federal debt outstanding (end of fiscal year)
11. Total federal debt held by the public (end of fiscal year)
12. Statutory debt limit status (suspended or specific amount)
13. Average length (in months) of marketable interest-bearing public debt
14. Per capita currency and coin in circulation
15. Exchange Stabilization Fund total assets
16. Exchange rate of Canadian dollars per U.S. dollar (weekly)

**Source files:** `data/officeqa/all_labels.json`, `data/officeqa/queries.json`,
`data/officeqa/text/`, `data/officeqa/json/`

---

## 5. Publications

**Description:** Academic / scientific publications (research papers), used for
extracting bibliographic metadata, authorship, and study details.

**Source / link:** Not recorded in the repo (e.g., arXiv https://arxiv.org or
PubMed https://pubmed.ncbi.nlm.nih.gov — confirm actual source).

**Common extractable values:**

1. Paper title
2. List of authors
3. Author affiliations and corresponding author
4. Publication venue (journal / conference) and volume/issue
5. Publication year / date
6. DOI or other persistent identifier
7. Abstract
8. Keywords / index terms
9. Research field / subject area
10. Funding sources / grant numbers
11. Number of references cited
12. Dataset(s) or code repositories referenced
13. Main contributions / reported results

**Source files:** `data/publications/`

---

## 6. Medical Records

**Description:** Clinical patient records (e.g., admission notes, discharge
summaries), used for extracting patient demographics, diagnoses, treatments, and
visit details.

**Source / link:** Not recorded in the repo (e.g., MIMIC-III/IV
https://physionet.org or a synthetic generator like Synthea
https://synthetichealth.github.io/synthea/ — confirm actual source).

**Common extractable values:**

1. Patient identifier / medical record number (MRN)
2. Patient age and sex
3. Admission date and discharge date
4. Chief complaint / reason for visit
5. Primary diagnosis and secondary diagnoses (with ICD codes)
6. Procedures performed (with codes)
7. Medications prescribed (name, dose, frequency)
8. Allergies
9. Vital signs (blood pressure, heart rate, temperature)
10. Lab results / test values
11. Attending physician / care provider
12. Department / service / facility
13. Discharge disposition and follow-up instructions

**Source files:** `data/medical/`

---

## 7. NHC Tropical Cyclone Reports (nhc_tcr)

**Description:** Post-storm reports published by the U.S. National Hurricane
Center, one per Atlantic / Eastern-Pacific cyclone — born-digital PDFs with a
highly consistent template (cover block → Synoptic History → Meteorological
Statistics → Casualty & Damage → Forecast Critique).

**Source / link:** National Hurricane Center Tropical Cyclone Reports —
https://www.nhc.noaa.gov/data/tcr/

**Common extractable values:**

1. Name of the tropical cyclone
2. Official basin-year identifier (e.g., `AL142024`)
3. Report issue date
4. Lead (first-listed) author of the report
5. Date and time (UTC) the cyclone first became a tropical depression
6. Geographic feature/region the precursor disturbance originated from
7. Landfall or closest-approach date
8. Estimated minimum central pressure
9. Estimated peak maximum sustained wind (knots)
10. Peak storm surge height
11. Number of direct deaths attributed to the cyclone
12. Total damage estimate (USD)
13. Quality of the genesis (formation) forecast
14. Whether intensity forecast errors were larger/smaller than the previous 5-year mean

**Source files:** `data/tropic/queries.json`, `data/tropic/docs/`

---

## 8. EMA EPAR Product Information (epar)

**Description:** The Annex I / Summary of Product Characteristics for
centrally-authorised EU medicines, published by the European Medicines Agency —
native PDFs with a legally-mandated numbered-section structure (§1 Name →
§2 Composition → §4 Clinical → §5 Pharmacology → §7–10 admin).

**Source / link:** EMA medicines (EPAR / Product Information) —
https://www.ema.europa.eu/en/medicines

**Common extractable values:**

1. Whether the product is subject to additional monitoring (black inverted triangle statement)
2. Name of the medicinal product (including strength and pharmaceutical form)
3. Active substance
4. Pharmaceutical form
5. Disease / therapeutic area named in the first therapeutic indication
6. Recommended dose for the first posology setting
7. First contraindication listed
8. Pharmacotherapeutic group
9. ATC code
10. Elimination / terminal half-life of the active substance
11. Storage conditions
12. Marketing Authorisation Holder
13. Date of first authorisation

**Source files:** `data/product/queries.json`, `data/product/docs/`
