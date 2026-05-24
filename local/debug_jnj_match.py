#!/usr/bin/env python3
"""Debug J&J matching."""
import importlib.util
import json
import re
from pathlib import Path

# Load rule
rule_path = Path("rules/agent/financebench_agent/what_is_total_assets_at_year_end__from_the_audited_balance_s/rule_table_total_assets_balance_sheet.py")
spec = importlib.util.spec_from_file_location("rule_mod", rule_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
rule_fn = mod.rule_table_total_assets_balance_sheet

# Load doc
path = Path("data/financebench/processing/JOHNSON_JOHNSON_2022_10K_reconstructed.json")
with open(path) as f:
    doc = json.load(f)

spans = rule_fn(doc)
retrieved_text = "\n".join(s.get("text", "") for s in spans)

print("Retrieved text (first 1000 chars):")
print(retrieved_text[:1000])
print()

# Find all numbers
retrieved_nums = re.findall(r'[\d,]+', retrieved_text)
print(f"Numbers found: {len(retrieved_nums)}")
large_nums = [n for n in retrieved_nums if len(n.replace(',', '')) >= 5]
print(f"Large numbers (5+ digits): {large_nums[:20]}")

# Target: 187.4 billion = 187,400 million
target = 187.4 * 1000
print(f"\nTarget value (millions): {target}")

for rnum in large_nums:
    try:
        rval = float(rnum.replace(',', ''))
        if rval > 100000:
            diff = abs(rval - target)
            pct = diff / target * 100
            print(f"  {rnum} = {rval}, diff from target: {diff:.0f} ({pct:.2f}%)")
    except:
        pass
