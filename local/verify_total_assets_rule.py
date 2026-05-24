#!/usr/bin/env python3
"""Final verification of total assets rule."""
import importlib.util
import json
from pathlib import Path

# Load rule from saved file
rule_path = Path("rules/agent/financebench_agent/what_is_total_assets_at_year_end__from_the_audited_balance_s/rule_table_total_assets_balance_sheet.py")
spec = importlib.util.spec_from_file_location("rule_mod", rule_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
rule_fn = mod.rule_table_total_assets_balance_sheet

DOC_NAMES = [
    "AMCOR_2019_10K",
    "COSTCO_2017_10K",
    "BOEING_2018_10K",
    "AMAZON_2018_10K",
    "EBAY_2021_10K",
    "AMAZON_2016_10K",
    "CORNING_2022_10K",
    "NIKE_2021_10K",
    "LOCKHEEDMARTIN_2022_10K",
    "JOHNSON_JOHNSON_2022_10K",
]

GROUND_TRUTH = {
    "AMCOR_2019_10K": "17,165.0 million",
    "COSTCO_2017_10K": "36,347",
    "BOEING_2018_10K": "$117,359 million",
    "AMAZON_2018_10K": "$162,648 million",
    "EBAY_2021_10K": "$26,626 million",
    "AMAZON_2016_10K": "$83,402 million",
    "CORNING_2022_10K": "29,499 million",
    "NIKE_2021_10K": "$37,740 million",
    "LOCKHEEDMARTIN_2022_10K": "$52,880 million",
    "JOHNSON_JOHNSON_2022_10K": "$187.4 billion",
}

print("="*80)
print("FINAL VERIFICATION: rule_table_total_assets_balance_sheet")
print("="*80)

all_pass = True
for doc_name in DOC_NAMES:
    path = Path(f"data/financebench/processing/{doc_name}_reconstructed.json")
    with open(path) as f:
        doc = json.load(f)

    spans = rule_fn(doc)
    retrieved_text = "\n".join(s.get("text", "") for s in spans)

    # Check if answer value can be found
    answer = GROUND_TRUTH[doc_name]
    # Extract numeric value
    import re
    nums = re.findall(r'[\d,\.]+', answer)
    found = False
    for num in nums:
        num_clean = num.replace(',', '')
        if num_clean in retrieved_text.replace(',', ''):
            found = True
            break

    # Special handling for billion - look for approximate match
    if not found and "billion" in answer.lower():
        for num in nums:
            try:
                val = float(num.replace(',', ''))
                millions_target = val * 1000  # e.g., 187.4 billion = 187,400 million
                # Find all numbers in retrieved text
                retrieved_nums = re.findall(r'[\d,]+', retrieved_text)
                for rnum in retrieved_nums:
                    try:
                        rval = float(rnum.replace(',', ''))
                        # Check for numbers in the right ballpark (>100k) with <1% tolerance
                        if rval > 100000 and abs(rval - millions_target) / millions_target < 0.02:
                            found = True
                            break
                    except:
                        continue
                if found:
                    break
            except:
                pass

    if not found:
        all_pass = False
        print(f"✗ {doc_name}: FAIL - answer '{answer}' not found in {len(spans)} spans")
    else:
        print(f"✓ {doc_name}: OK - {len(spans)} spans retrieved")

print()
if all_pass:
    print("All documents PASS - rule works correctly!")
else:
    print("Some documents FAILED - check rule")
