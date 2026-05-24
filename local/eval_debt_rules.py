"""Evaluate long-term debt rules using LLM-as-a-judge."""

import json
import importlib.util
import sys
import time
from pathlib import Path

# Add src to path for model imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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
    """Apply all rules and return union of spans."""
    all_spans = []
    seen_ids = set()
    for rule_func in rules.values():
        for span in rule_func(doc):
            span_id = id(span)
            if span_id not in seen_ids:
                seen_ids.add(span_id)
                all_spans.append(span)
    return all_spans

def get_llm_answer(retrieved_text, question, model_mod):
    """Call LLM to answer the question based on retrieved text."""
    system_prompt = (
        "You are a financial document QA assistant.\n"
        "You are given a passage extracted from a financial filing and a question.\n"
        "Answer the question using only the provided passage.\n"
        'If the passage does not contain enough information to answer, reply with "NOT FOUND".\n'
        "Return only the answer — a short value or phrase, not a full sentence."
    )
    user_prompt = f"Passage:\n{retrieved_text}\n\nQuestion: {question}"

    response = model_mod.client.chat.completions.create(
        model=model_mod.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=500,
        temperature=0.0,
    )

    usage = response.usage
    return {
        "answer": (response.choices[0].message.content or "").strip(),
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }

def judge_answer(question, predicted, ground_truth, model_mod):
    """Use LLM to judge if predicted answer is correct."""
    system_prompt = """\
You are an answer equivalence judge for a financial document QA system.
You will be given a question, a predicted answer, and a ground truth answer.
Judge whether the predicted answer is correct — meaning semantically equivalent
to the ground truth, ignoring minor formatting differences.

Equivalence rules:
- Treat "2017" and "year 2017" as the same
- Treat "$4.5 billion" and "4,500 million" as the same if numerically equal
- Treat "NYSE" and "New York Stock Exchange" as the same
- Ignore leading/trailing whitespace, punctuation, and capitalization differences
- If the predicted answer is "NOT FOUND" or null, always judge as incorrect

Reply with exactly one word: CORRECT or INCORRECT"""

    user_prompt = (
        f"Question: {question}\n"
        f"Ground Truth: {ground_truth}\n"
        f"Predicted: {predicted}"
    )

    response = model_mod.client.chat.completions.create(
        model=model_mod.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=10,
        temperature=0.0,
    )

    verdict = (response.choices[0].message.content or "").strip().lower()
    usage = response.usage

    return {
        "correct": verdict == "correct",
        "verdict": verdict,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }

def main():
    print("Long-Term Debt Rules - LLM Judge Evaluation")
    print("=" * 60)

    # Track token usage
    total_qa_input = 0
    total_qa_output = 0
    total_judge_input = 0
    total_judge_output = 0
    total_llm_calls = 0

    start_time = time.time()

    # Load model
    try:
        import models.gpt54 as model_mod
        print("Using gpt54 model")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load rules
    rules = load_rules()
    print(f"Loaded {len(rules)} rules: {list(rules.keys())}")

    results = []
    correct_count = 0
    total_cost = 0

    for doc_name in DOC_NAMES:
        print(f"\n{doc_name}:")

        # Load document
        doc = load_doc(doc_name)

        # Apply rules
        spans = apply_rules(doc, rules)
        retrieved_text = "\n\n".join(s.get("text", "") for s in spans)

        # Calculate cost
        full_text = "\n".join(s.get("text", "") for s in doc.get("texts", []))
        full_tokens = count_tokens(full_text)
        retrieved_tokens = count_tokens(retrieved_text)
        cost = retrieved_tokens / full_tokens if full_tokens > 0 else 0
        total_cost += cost

        print(f"  Retrieved {len(spans)} spans, {retrieved_tokens} tokens, cost={cost:.4f}")

        # Get LLM answer
        qa_result = get_llm_answer(retrieved_text, QUESTION, model_mod)
        predicted = qa_result["answer"]
        total_qa_input += qa_result["input_tokens"]
        total_qa_output += qa_result["output_tokens"]
        total_llm_calls += 1

        print(f"  Predicted: {predicted}")

        # Judge answer
        ground_truth = GROUND_TRUTH[doc_name]
        judge_result = judge_answer(QUESTION, predicted, ground_truth, model_mod)
        total_judge_input += judge_result["input_tokens"]
        total_judge_output += judge_result["output_tokens"]
        total_llm_calls += 1

        if judge_result["correct"]:
            correct_count += 1
            print(f"  ✓ CORRECT (ground truth: {ground_truth})")
        else:
            print(f"  ✗ INCORRECT (ground truth: {ground_truth})")

        results.append({
            "doc_name": doc_name,
            "predicted": predicted,
            "ground_truth": ground_truth,
            "correct": judge_result["correct"],
            "cost_ratio": cost,
            "num_spans": len(spans),
            "retrieved_tokens": retrieved_tokens,
        })

    # Summary
    elapsed = time.time() - start_time
    accuracy = correct_count / len(DOC_NAMES)
    avg_cost = total_cost / len(DOC_NAMES)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Merge Accuracy: {correct_count}/{len(DOC_NAMES)} = {accuracy:.2%}")
    print(f"Avg Cost Ratio: {avg_cost:.4f}")
    print(f"Total LLM Calls: {total_llm_calls}")
    print(f"Total Input Tokens: {total_qa_input + total_judge_input}")
    print(f"Total Output Tokens: {total_qa_output + total_judge_output}")
    print(f"Elapsed Time: {elapsed:.2f}s")

    # Save results
    output = {
        "question": QUESTION,
        "question_slug": QUESTION_SLUG,
        "merge_accuracy": accuracy,
        "avg_cost_ratio": avg_cost,
        "total_llm_calls": total_llm_calls,
        "total_input_tokens": total_qa_input + total_judge_input,
        "total_output_tokens": total_qa_output + total_judge_output,
        "elapsed_seconds": elapsed,
        "per_document": results,
    }

    output_path = Path(f"local/eval_debt_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return output

if __name__ == "__main__":
    main()
