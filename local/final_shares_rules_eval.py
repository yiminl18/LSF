#!/usr/bin/env python3
"""Final evaluation of shares outstanding rules (substring match proxy)."""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tiktoken

# Configuration
QUESTION = "How many shares of common stock were outstanding as of the cover-page reference date?"
QUESTION_SLUG = "how_many_shares_of_common_stock_were_outstanding_as_of_the_c"

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

RULES_DIR = Path("rules/agent/financebench_agent") / QUESTION_SLUG
LABELS_FILE = Path("data/financebench/sample_doc_labels.json")
PROCESSING_DIR = Path("data/financebench/processing")

start_time = time.time()
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def load_rule_fn(rule_file: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_rule_mod", str(rule_file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, obj in vars(mod).items():
        if name.startswith("rule_") and callable(obj):
            return obj
    raise ValueError(f"No rule_* function found in {rule_file}")

def main():
    labels = json.loads(LABELS_FILE.read_text())
    ground_truth = {}
    for doc_name in DOC_NAMES:
        doc_key = f"{doc_name}.pdf"
        if doc_key in labels:
            ground_truth[doc_name] = labels[doc_key][QUESTION]

    rule_files = sorted(RULES_DIR.glob("*.py"))
    rules = [(rf.stem, load_rule_fn(rf)) for rf in rule_files]

    print(f"Loaded {len(rules)} rules:")
    for name, _ in rules:
        print(f"  - {name}")

    results = []
    correct_count = 0
    total_cost = 0.0
    rule_coverage = {name: 0 for name, _ in rules}
    rule_costs = {name: [] for name, _ in rules}

    print("\n" + "="*70)
    print("Evaluating (substring match)")
    print("="*70 + "\n")

    for doc_name in DOC_NAMES:
        doc_path = PROCESSING_DIR / f"{doc_name}_reconstructed.json"
        doc = json.loads(doc_path.read_text())
        gt = ground_truth[doc_name]

        all_spans = []
        for rule_name, rule_fn in rules:
            spans = rule_fn(doc)
            if spans:
                rule_coverage[rule_name] += 1
                all_spans.extend(spans)

        seen = set()
        unique_spans = []
        for s in all_spans:
            key = s.get("text", "")
            if key not in seen:
                seen.add(key)
                unique_spans.append(s)

        retrieved_text = " ".join(s.get("text", "") for s in unique_spans)
        doc_text = " ".join(s.get("text", "") for s in doc.get("texts", []))

        retrieved_tokens = count_tokens(retrieved_text)
        doc_tokens = count_tokens(doc_text)
        cost_ratio = retrieved_tokens / doc_tokens if doc_tokens > 0 else 0
        total_cost += cost_ratio

        for rule_name, rule_fn in rules:
            spans = rule_fn(doc)
            if spans:
                rule_text = " ".join(s.get("text", "") for s in spans)
                rule_tokens = count_tokens(rule_text)
                rule_costs[rule_name].append(rule_tokens / doc_tokens if doc_tokens > 0 else 0)

        gt_clean = gt.replace(",", "")
        hit = gt_clean in retrieved_text.replace(",", "")

        if hit:
            correct_count += 1

        status = "HIT" if hit else "MISS"
        print(f"{doc_name}: {status}")
        print(f"  Ground truth: {gt}")
        print(f"  Cost ratio: {cost_ratio:.6f}")
        print()

        results.append({
            "doc_name": doc_name,
            "ground_truth": gt,
            "hit": hit,
            "cost_ratio": cost_ratio,
            "retrieved_tokens": retrieved_tokens,
            "doc_tokens": doc_tokens,
        })

    merge_accuracy = correct_count / len(DOC_NAMES)
    avg_cost = total_cost / len(DOC_NAMES)
    latency = time.time() - start_time

    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Merge accuracy (substring): {merge_accuracy:.2%} ({correct_count}/{len(DOC_NAMES)})")
    print(f"Average cost ratio: {avg_cost:.6f}")
    print(f"Latency: {latency:.2f}s")
    print()
    print("Per-rule statistics:")
    for rule_name, _ in rules:
        coverage = rule_coverage[rule_name]
        costs = rule_costs[rule_name]
        avg_rule_cost = sum(costs) / len(costs) if costs else 0
        print(f"  {rule_name}: coverage={coverage}/{len(DOC_NAMES)}, avg_cost={avg_rule_cost:.6f}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_data = {
        "question": QUESTION,
        "question_slug": QUESTION_SLUG,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_seconds": round(latency, 2),
        "agent_input_tokens": 0,
        "agent_output_tokens": 0,
        "total_llm_calls": 0,
        "num_rules": len(rules),
        "merge_accuracy": round(merge_accuracy, 4),
        "avg_cost_ratio": round(avg_cost, 6),
        "rules": [
            {
                "rule_name": rule_name,
                "description": rule_fn.__doc__ or "",
                "coverage": rule_coverage[rule_name],
                "avg_cost_ratio": round(sum(rule_costs[rule_name]) / len(rule_costs[rule_name]), 6) if rule_costs[rule_name] else 0,
                "file": str(RULES_DIR / f"{rule_name}.py"),
            }
            for rule_name, rule_fn in rules
        ],
        "per_document": results,
    }

    log_file = RULES_DIR.parent / f"{QUESTION_SLUG}_claude-opus-4-5_{timestamp}_rule_gen.json"
    log_file.write_text(json.dumps(log_data, indent=2, ensure_ascii=False))
    print(f"\nLog saved to: {log_file}")

    return merge_accuracy, avg_cost

if __name__ == "__main__":
    accuracy, cost = main()
    print(f"\nFinal: accuracy={accuracy:.2%}, avg_cost={cost:.6f}")
