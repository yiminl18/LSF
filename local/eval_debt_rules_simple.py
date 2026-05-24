"""Evaluate long-term debt rules using sophisticated string matching."""

import json
import importlib.util
import re
import time
from pathlib import Path

QUESTION = "What is long-term debt at year-end (0 if none)?"
QUESTION_SLUG = "what_is_long_term_debt_at_year_end__0_if_none"

GROUND_TRUTH = {
    "AMCOR_2019_10K": "5,314.4 million",
    "COSTCO_2017_10K": "6,573",
    "BOEING_2018_10K": "$10,657 million",
    "AMAZON_2018_10K": "$50,708 million",
    "EBAY_2021_10K": "$7,727 million",
    "AMAZON_2016_10K": "$7,694 million",
    "CORNING_2022_10K": "6,687 million",
    "NIKE_2021_10K": "$9,413 million",
    "LOCKHEEDMARTIN_2022_10K": "$15,547 million",
    "JOHNSON_JOHNSON_2022_10K": "$26.9 billion",
}

# Normalized values (in millions) for verification
NORMALIZED_VALUES = {
    "AMCOR_2019_10K": 5314.4,
    "COSTCO_2017_10K": 6573,
    "BOEING_2018_10K": 10657,
    "AMAZON_2018_10K": 50708,
    "EBAY_2021_10K": 7727,
    "AMAZON_2016_10K": 7694,
    "CORNING_2022_10K": 6687,
    "NIKE_2021_10K": 9413,
    "LOCKHEEDMARTIN_2022_10K": 15547,
    "JOHNSON_JOHNSON_2022_10K": 26888,  # 26.9 billion = 26,900 million, but actual is 26,888
}

DOC_NAMES = list(GROUND_TRUTH.keys())

def count_tokens(text):
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except:
        return len(text) // 4

def load_doc(doc_name):
    path = Path(f"data/financebench/processing/{doc_name}_reconstructed.json")
    with open(path) as f:
        return json.load(f)

def load_rules():
    rule_dir = Path("rules/agent/financebench_agent/what_is_long_term_debt_at_year_end__0_if_none")
    rules = {}
    for rule_file in rule_dir.glob("rule_*.py"):
        rule_name = rule_file.stem
        spec = importlib.util.spec_from_file_location(rule_name, rule_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rules[rule_name] = getattr(module, rule_name)
    return rules

def apply_rules(doc, rules):
    all_spans = []
    seen_ids = set()
    for rule_func in rules.values():
        for span in rule_func(doc):
            span_id = id(span)
            if span_id not in seen_ids:
                seen_ids.add(span_id)
                all_spans.append(span)
    return all_spans

def extract_number(text):
    """Extract numeric value from text, handling various formats."""
    # Remove $ and other currency symbols
    text = text.replace("$", "").replace("€", "").replace("£", "")

    # Handle billions
    if "billion" in text.lower():
        match = re.search(r"[\d,]+\.?\d*", text)
        if match:
            num = float(match.group().replace(",", ""))
            return num * 1000  # Convert to millions

    # Handle millions
    match = re.search(r"[\d,]+\.?\d*", text)
    if match:
        return float(match.group().replace(",", ""))

    return None

def check_value_in_text(text, expected_value):
    """Check if the expected value appears in text with various formats."""
    # Normalize expected value
    exp_str = str(expected_value).replace(",", "")

    # Also check for slight variations
    text_normalized = text.replace(",", "").replace("$", "")

    if exp_str in text_normalized:
        return True

    # Check with some tolerance for decimal variations
    if isinstance(expected_value, float):
        int_val = int(expected_value)
        if str(int_val) in text_normalized:
            return True

    return False

def find_long_term_debt_value(spans, expected_value):
    """Try to extract the long-term debt value from retrieved spans."""
    for span in spans:
        text = span.get("text", "")

        # Look for table cells with long-term debt row header
        if span.get("label") == "table":
            cells = span.get("table_data", {}).get("cells", [])

            # Find rows with long-term debt header
            for cell in cells:
                if cell.get("is_row_header"):
                    header_text = cell.get("text", "").lower()
                    if "long-term debt" in header_text or "long term debt" in header_text:
                        row = cell.get("row")
                        # Find values in same row
                        for value_cell in cells:
                            if value_cell.get("row") == row and not value_cell.get("is_row_header"):
                                cell_val = extract_number(value_cell.get("text", ""))
                                if cell_val and abs(cell_val - expected_value) < expected_value * 0.1:
                                    return True

                    # Also check for "long-term obligations" (Amazon case)
                    if "long-term obligations" in header_text:
                        row = cell.get("row")
                        for value_cell in cells:
                            if value_cell.get("row") == row and not value_cell.get("is_row_header"):
                                cell_val = extract_number(value_cell.get("text", ""))
                                if cell_val and abs(cell_val - expected_value) < expected_value * 0.1:
                                    return True

        # Simple text check as fallback
        if check_value_in_text(text, expected_value):
            return True

    return False

def main():
    print("Long-Term Debt Rules - Simple Evaluation")
    print("=" * 60)

    start_time = time.time()

    rules = load_rules()
    print(f"Loaded {len(rules)} rules: {list(rules.keys())}")

    results = []
    correct_count = 0
    total_cost = 0

    for doc_name in DOC_NAMES:
        print(f"\n{doc_name}:")

        doc = load_doc(doc_name)
        spans = apply_rules(doc, rules)
        retrieved_text = "\n\n".join(s.get("text", "") for s in spans)

        full_text = "\n".join(s.get("text", "") for s in doc.get("texts", []))
        full_tokens = count_tokens(full_text)
        retrieved_tokens = count_tokens(retrieved_text)
        cost = retrieved_tokens / full_tokens if full_tokens > 0 else 0
        total_cost += cost

        print(f"  Retrieved {len(spans)} spans, {retrieved_tokens} tokens, cost={cost:.4f}")

        # Check if value is found
        expected = NORMALIZED_VALUES[doc_name]
        found = find_long_term_debt_value(spans, expected)

        ground_truth = GROUND_TRUTH[doc_name]

        if found:
            correct_count += 1
            print(f"  ✓ FOUND value ~{expected} (ground truth: {ground_truth})")
        else:
            print(f"  ✗ NOT FOUND value ~{expected} (ground truth: {ground_truth})")

        results.append({
            "doc_name": doc_name,
            "expected_value": expected,
            "ground_truth": ground_truth,
            "found": found,
            "cost_ratio": cost,
            "num_spans": len(spans),
            "retrieved_tokens": retrieved_tokens,
        })

    elapsed = time.time() - start_time
    accuracy = correct_count / len(DOC_NAMES)
    avg_cost = total_cost / len(DOC_NAMES)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Value Detection Accuracy: {correct_count}/{len(DOC_NAMES)} = {accuracy:.2%}")
    print(f"Avg Cost Ratio: {avg_cost:.4f}")
    print(f"Elapsed Time: {elapsed:.2f}s")

    output = {
        "question": QUESTION,
        "question_slug": QUESTION_SLUG,
        "accuracy": accuracy,
        "avg_cost_ratio": avg_cost,
        "elapsed_seconds": elapsed,
        "per_document": results,
    }

    output_path = Path(f"local/eval_debt_results_simple.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return output

if __name__ == "__main__":
    main()
