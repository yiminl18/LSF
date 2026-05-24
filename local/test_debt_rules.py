"""Test long-term debt rules against sampled documents."""

import json
import sys
from pathlib import Path

# Add rules directory to path
sys.path.insert(0, str(Path("rules/agent/financebench_agent/what_is_long_term_debt_at_year_end__0_if_none")))

GROUND_TRUTH = {
    "AMCOR_2019_10K": "5,314.4",
    "COSTCO_2017_10K": "6,573",
    "BOEING_2018_10K": "10,657",
    "AMAZON_2018_10K": "50,708",
    "EBAY_2021_10K": "7,727",
    "AMAZON_2016_10K": "7,694",
    "CORNING_2022_10K": "6,687",
    "NIKE_2021_10K": "9,413",
    "LOCKHEEDMARTIN_2022_10K": "15,547",
    "JOHNSON_JOHNSON_2022_10K": "26,888",
}

def load_doc(doc_name):
    path = Path(f"data/financebench/processing/{doc_name}_reconstructed.json")
    with open(path) as f:
        return json.load(f)

def normalize_number(s):
    return s.replace(",", "").replace("$", "").strip()

def count_tokens(text):
    """Rough token count using tiktoken-like estimate."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except:
        return len(text) // 4  # fallback

def check_hit(spans, expected_value):
    """Check if any span contains the expected value."""
    norm_value = normalize_number(expected_value)
    for span in spans:
        text = span.get("text", "").replace(",", "")
        if norm_value in text:
            return True
    return False

def get_full_doc_text(doc):
    """Get full document text."""
    return "\n".join(span.get("text", "") for span in doc.get("texts", []))

def get_spans_text(spans):
    """Get concatenated text from spans."""
    return "\n".join(span.get("text", "") for span in spans)

def load_rule(rule_name):
    """Load and return a rule function."""
    import importlib.util
    rule_path = Path(f"rules/agent/financebench_agent/what_is_long_term_debt_at_year_end__0_if_none/{rule_name}.py")
    spec = importlib.util.spec_from_file_location(rule_name, rule_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, rule_name)

def test_rule(rule_func, rule_name):
    """Test a single rule against all documents."""
    print(f"\nTesting {rule_name}:")
    print("-" * 60)

    hits = 0
    total_cost = 0

    for doc_name, expected in GROUND_TRUTH.items():
        doc = load_doc(doc_name)
        spans = rule_func(doc)

        # Check hit
        hit = check_hit(spans, expected)
        if hit:
            hits += 1

        # Calculate cost
        full_tokens = count_tokens(get_full_doc_text(doc))
        span_tokens = count_tokens(get_spans_text(spans))
        cost = span_tokens / full_tokens if full_tokens > 0 else 0
        total_cost += cost

        status = "✓" if hit else "✗"
        print(f"  {status} {doc_name}: {len(spans)} spans, cost={cost:.4f}")

    avg_cost = total_cost / len(GROUND_TRUTH)
    hit_rate = hits / len(GROUND_TRUTH)

    print(f"\n  Hit rate: {hits}/{len(GROUND_TRUTH)} = {hit_rate:.2%}")
    print(f"  Avg cost: {avg_cost:.4f}")

    return hits, avg_cost

def test_all_rules():
    """Test all rules and combined results."""
    rule_dir = Path("rules/agent/financebench_agent/what_is_long_term_debt_at_year_end__0_if_none")
    rule_files = list(rule_dir.glob("rule_*.py"))

    print(f"Found {len(rule_files)} rules")

    rules = {}
    for rule_file in rule_files:
        rule_name = rule_file.stem
        rules[rule_name] = load_rule(rule_name)

    # Test individual rules
    for rule_name, rule_func in rules.items():
        test_rule(rule_func, rule_name)

    # Test combined rules (union)
    print("\n" + "=" * 60)
    print("COMBINED RULES (union):")
    print("=" * 60)

    combined_hits = 0
    combined_total_cost = 0

    for doc_name, expected in GROUND_TRUTH.items():
        doc = load_doc(doc_name)

        # Get union of all spans
        all_spans = []
        seen_ids = set()
        for rule_func in rules.values():
            for span in rule_func(doc):
                span_id = id(span)
                if span_id not in seen_ids:
                    seen_ids.add(span_id)
                    all_spans.append(span)

        # Check hit
        hit = check_hit(all_spans, expected)
        if hit:
            combined_hits += 1

        # Calculate cost
        full_tokens = count_tokens(get_full_doc_text(doc))
        span_tokens = count_tokens(get_spans_text(all_spans))
        cost = span_tokens / full_tokens if full_tokens > 0 else 0
        combined_total_cost += cost

        status = "✓" if hit else "✗"
        print(f"  {status} {doc_name}: {len(all_spans)} spans, cost={cost:.4f}")

    avg_cost = combined_total_cost / len(GROUND_TRUTH)
    hit_rate = combined_hits / len(GROUND_TRUTH)

    print(f"\n  Combined hit rate: {combined_hits}/{len(GROUND_TRUTH)} = {hit_rate:.2%}")
    print(f"  Combined avg cost: {avg_cost:.4f}")

if __name__ == "__main__":
    test_all_rules()
