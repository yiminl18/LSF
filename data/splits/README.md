# Train/Test Split Documentation

## Current Split: `train_test_split_big.json`

### Overview
- **Strategy**: Company holdout (no company appears in both train and test)
- **Total PDFs**: 212 (all PDFs with answerable ground truth)
- **Train/Test Ratio**: 8:2 (170:42)
- **Seed**: 42 (reproducible)
- **Created**: 2025-01-02

### Statistics

| Set | PDFs | Companies | % of Total |
|-----|------|-----------|------------|
| Train | 170 | 31 | 80.2% |
| Test | 42 | 6 | 19.8% |
| **Total** | **212** | **37** | **100%** |

### Test Companies (Holdout)
These 6 companies are **completely unseen** during training:

1. **AMCOR** (8 PDFs)
   - AMCOR_2019_10K, AMCOR_2021_10K, AMCOR_2022_10K, AMCOR_2022_8K_2022-07-01, 
     AMCOR_2022_8K_dated-2022-07-01, AMCOR_2023_10K, AMCOR_2023Q2_10Q, 
     AMCOR_2023Q4_EARNINGS

2. **AMD** (5 PDFs)
   - AMD_2019_10K, AMD_2020_10K, AMD_2021_10K, AMD_2022_10K, AMD_2022_annualreport

3. **BESTBUY** (10 PDFs)
   - BESTBUY_2015_10K, BESTBUY_2020_10K, BESTBUY_2021_10K, BESTBUY_2022_10K, 
     BESTBUY_2023_10K, BESTBUY_2023_8K_dated-2023-04-12, BESTBUY_2023_8K_dated-2023-04-24, 
     BESTBUY_2023Q4_EARNINGS, BESTBUY_2024Q2_10Q, BESTBUY_2024Q2_EARNINGS

4. **KRAFTHEINZ** (6 PDFs)
   - KRAFTHEINZ_2015_10K, KRAFTHEINZ_2016_10K, KRAFTHEINZ_2019_10K, 
     KRAFTHEINZ_2020_10K, KRAFTHEINZ_2021_10K

5. **MCDONALDS** (7 PDFs)
   - MCDONALDS_2022_10K, MCDONALDS_2023Q1_EARNINGS, MCDONALDS_2023Q2_EARNINGS, 
     MCDONALDS_8K_dated-2023-01-06, MCDONALDS_8K_dated-2023-02-13, MCDONALDS_8K_dated-2023-08-14

6. **NIKE** (8 PDFs)
   - NIKE_2016_10K, NIKE_2017_10K, NIKE_2018_10K, NIKE_2019_10K, 
     NIKE_2020_10K, NIKE_2021_10K, NIKE_2022_10K, NIKE_2023_10K

### Train Companies (31)
ACTIVISIONBLIZZARD, ADOBE, AMAZON, AMERICANWATERWORKS, APPLE, BLOCK, BOEING, 
CORNING, COSTCO, EBAY, FEDEX, FOOTLOCKER, GENERALMILLS, INTEL, JOHNSON, 
LOCKHEEDMARTIN, MGMRESORTS, MICROSOFT, NETFLIX, ORACLE, PAYPAL, PEPSICO, 
PG, Pfizer, SALESFORCE, ULTABEAUTY, VERIZON, WALMART, 3M, ACTIVSIONBLIZZARD

### Document Type Distribution

#### Test Set (42 PDFs)
- **10K**: 27 (64.3%)
- **8K**: 7 (16.7%)
- **EARNINGS**: 5 (11.9%)
- **10Q**: 2 (4.8%)
- **annualreport**: 1 (2.4%)

#### Train Set (170 PDFs)
- **10K**: 103 (60.6%)
- **EARNINGS**: 23 (13.5%)
- **8K**: 21 (12.4%)
- **10Q**: 17 (10.0%)
- **annualreport**: 4 (2.4%)
- **Other**: 2 (1.2%)

✅ Both sets have good coverage of all document types.

---

## Rationale

### Why Company Holdout?
- **Prevent leakage**: Company name, layout style, writing patterns are company-specific
- **Test generalization**: Can the model work on completely new companies?
- **Realistic scenario**: In production, new companies will appear

### Why 8:2 Ratio?
- **Sufficient training data**: 170 PDFs × 30 questions = 5,100 potential training instances
- **Sufficient test data**: 42 PDFs × 30 questions = 1,260 test instances
- **Statistical significance**: 42 PDFs is enough for stable recall@k metrics

### Why Balance by Doc Type?
- Different document types have different layouts (10K vs 8K vs EARNINGS)
- Ensures test set isn't biased toward one type
- Makes metrics more representative

---

## Usage

### In Python
```python
import json
from pathlib import Path

# Load split
with open('data/splits/train_test_split_big.json') as f:
    split = json.load(f)

train_ids = split['train_pdf_ids']
test_ids = split['test_pdf_ids']

print(f"Train: {len(train_ids)} PDFs")
print(f"Test: {len(test_ids)} PDFs")
```

### In Experiments
All scripts that use train/test split should point to this file:
```bash
python test/run_similarity_q1.py \
  --split data/splits/train_test_split_big.json \
  --question-index 11
```

---

## Regenerating Split

If you need a different split (e.g., 7:3, different seed):

```bash
python pipeline/utils/make_company_holdout_split.py \
  --test-company-frac 0.3 \
  --min-test-pdfs 60 \
  --seed 123 \
  --out data/splits/train_test_split_70_30.json
```

---

## Legacy Split

The old split (`train_test_split.json`) is preserved:
- 30 train PDFs, 10 test PDFs
- Used for initial experiments (q1-10)
- Can be found at: `data/splits/train_test_split.json` (copied from root)

---

## Notes

- **No overlap**: A PDF ID can only appear in train OR test, never both
- **Company consistency**: All PDFs from a company are in the same set
- **Deterministic**: Same seed always produces same split
- **Answerable GT only**: Only PDFs with at least one answerable question included

Last updated: 2025-01-02


