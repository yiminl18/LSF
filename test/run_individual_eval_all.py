"""Run individual rule evaluation for all 10 sample questions, both splits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
import os
os.chdir(_ROOT)

from rule_apply_individual import rule_apply_individual
from eval_rule import eval_rule

QUESTIONS = [
    ("What is long-term debt at year-end (0 if none)?",
     "what_is_longterm_debt_at_yearend_0_if_none_10"),
    ("What is net income (loss) for the most recent fiscal year?",
     "what_is_net_income_loss_for_the_most_recent_fiscal_year_10"),
    ("What is total revenue for the most recent fiscal year (from the audited income statement)?",
     "what_is_total_revenue_for_the_most_recent_fiscal_year_from_t_10"),
    ("What is total assets at year-end (from the audited balance sheet)?",
     "what_is_total_assets_at_yearend_from_the_audited_balance_she_10"),
    ("What is the registrant's telephone number?",
     "what_is_the_registrants_telephone_number_10"),
    ("How many shares of common stock were outstanding as of the cover-page reference date?",
     "how_many_shares_of_common_stock_were_outstanding_as_of_the_c_10"),
    ("What is the registrant's exact name?",
     "what_is_the_registrants_exact_name_10"),
    ("What is the state (or other jurisdiction) of incorporation and the IRS Employer Identification Number?",
     "what_is_the_state_or_other_jurisdiction_of_incorporation_and_10"),
    ("What is the address of principal executive offices and ZIP code?",
     "what_is_the_address_of_principal_executive_offices_and_zip_c_10"),
    ("What is/are the trading symbol(s) and listing exchange(s)?",
     "what_isare_the_trading_symbols_and_listing_exchanges_10"),
]

SPLITS = [
    ("sampled", "data/financebench/sample/single_cluster/random/sample_doc_labels.json", "results/financebench_single_cluster/llm/gpt54/one_shot/eval_individual"),
]

CORRECT_LABELS = "data/financebench/correct_labels.json"
PROCESSING_DIR = "data/financebench/processing"
RULES_BASE     = _ROOT / "rules/financebench_single_cluster/llm/gpt54/one_shot"
RULE_RUN_DIR   = "results/financebench_single_cluster/llm/gpt54/one_shot/rule_run_individual"


def get_doc_names(labels_file: str) -> list[str]:
    data = json.loads(Path(labels_file).read_text(encoding="utf-8"))
    return [k.removesuffix(".pdf") for k in data.keys()]


for question, slug in QUESTIONS:
    rule_files = sorted((RULES_BASE / slug).glob("rule_*.py"))
    print(f"\n{'='*70}")
    print(f"Question: {question}")
    print(f"Slug:     {slug}")
    print(f"Rules:    {len(rule_files)}")

    for split_tag, labels_file, eval_out_dir in SPLITS:
        doc_names = get_doc_names(labels_file)
        print(f"\n  Split: {split_tag} ({len(doc_names)} docs)")

        # Step 1 — rule_apply_individual (skip already-done docs per rule)
        for rf in rule_files:
            rule_name = rf.stem
            pred_path = _ROOT / RULE_RUN_DIR / slug / f"{rule_name}_individual.json"
            done: set[str] = set()
            if pred_path.exists():
                done = {r["doc_name"] for r in json.loads(pred_path.read_text(encoding="utf-8"))}
            missing = [d for d in doc_names if d not in done]
            if not missing:
                continue
            print(f"    rule_apply {rule_name}: {len(missing)} docs remaining")
            for doc_name in missing:
                doc_path = _ROOT / PROCESSING_DIR / f"{doc_name}_reconstructed.json"
                if not doc_path.exists():
                    print(f"      skip {doc_name}: no JSON")
                    continue
                doc = json.loads(doc_path.read_text(encoding="utf-8"))
                try:
                    rule_apply_individual(
                        document=doc,
                        rule_name=rule_name,
                        question_slug=slug,
                        question=question,
                        rules_dir="rules/financebench_single_cluster/llm/gpt54/one_shot",
                        output_dir=RULE_RUN_DIR,
                    )
                except Exception as exc:
                    print(f"      error on {doc_name}: {exc}")

        # Step 2 — eval_rule (skip if eval file already exists)
        rule_results: list[dict] = []
        for rf in rule_files:
            rule_name = rf.stem
            eval_path = Path(eval_out_dir) / slug / f"{rule_name}_eval.json"
            if eval_path.exists():
                result = json.loads(eval_path.read_text(encoding="utf-8"))
                print(f"    eval {rule_name}: cached (acc={result['accuracy']:.2f})")
            else:
                print(f"    eval {rule_name} ...")
                try:
                    result = eval_rule(
                        rule_name=rule_name,
                        doc_names=doc_names,
                        question=question,
                        question_slug=slug,
                        labels_file=CORRECT_LABELS,
                        processing_dir=PROCESSING_DIR,
                        output_dir=eval_out_dir,
                    )
                    print(f"      acc={result['accuracy']:.2f}  cost={result['avg_cost_ratio']:.5f}")
                except Exception as exc:
                    print(f"      ERROR: {exc}")
                    continue
            rule_results.append(result)

        # Step 3 — per-question summary sorted by accuracy desc
        rule_results.sort(key=lambda r: (-r["accuracy"], r["avg_cost_ratio"]))
        summary = {
            "question": question,
            "question_slug": slug,
            "split": split_tag,
            "num_documents": len(doc_names),
            "rules": [
                {
                    "rule_name": r["rule_name"],
                    "accuracy": r["accuracy"],
                    "avg_cost_ratio": r["avg_cost_ratio"],
                    "avg_rule_apply_latency_seconds": r.get("avg_rule_apply_latency_seconds"),
                    "avg_judge_latency_seconds": r.get("avg_judge_latency_seconds"),
                    "num_documents": r.get("num_documents"),
                }
                for r in rule_results
            ],
        }
        summary_path = Path(eval_out_dir) / slug / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Saved summary: {summary_path}")

# Final aggregate across all questions and splits
print("\n\nBuilding final aggregate summary...")
aggregate = []
for question, slug in QUESTIONS:
    entry = {"question": question, "question_slug": slug}
    for split_tag, _, eval_out_dir in SPLITS:
        summary_path = Path(eval_out_dir) / slug / "summary.json"
        if summary_path.exists():
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            rules = s.get("rules", [])
            if rules:
                best = rules[0]
                entry[split_tag] = {
                    "best_rule": best["rule_name"],
                    "best_accuracy": best["accuracy"],
                    "best_cost_ratio": best["avg_cost_ratio"],
                    "num_rules": len(rules),
                }
    aggregate.append(entry)

agg_path = Path("results/financebench_single_cluster/llm/gpt54/one_shot/eval_individual/summary.json")
agg_path.parent.mkdir(parents=True, exist_ok=True)
agg_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved aggregate: {agg_path}")
print("Done.")
