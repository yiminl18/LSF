import os, json, glob
from typing import Dict, List, Tuple
from gpt_4o_azure import gpt_4o_azure
import tiktoken


RESULTS_PATH = "/Users/evier/PycharmProjects/DocumentSplit/results.jsonl"
QUESTIONS_PATH = "/Users/evier/PycharmProjects/DocumentSplit/10k_factual_questions.txt"
GROUND_TRUTH_DIR = "/Users/evier/PycharmProjects/DocumentSplit/llm_gpt_4o_azure"
OUTPUT_SUMMARY_JSON = "/Users/evier/PycharmProjects/DocumentSplit/eval_summary.json"
OUTPUT_DETAIL_JSONL = "/Users/evier/PycharmProjects/DocumentSplit/eval_pairs.jsonl"


def load_questions(path: str) -> List[str]:
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_ground_truth(gt_dir: str) -> Dict[str, List[str]]:
    """
    Load ground-truth answers.
    Expected files: <doc_name>_answers.json with an array (len=30) or a dict mapping.
    Returns: { doc_name: [ans0, ans1, ..., ans29] }
    """
    mapping: Dict[str, List[str]] = {}
    for path in glob.glob(os.path.join(gt_dir, "*_answers.json")):
        filename = os.path.basename(path)
        # derive document name by stripping the suffix
        if filename.endswith("_answers.json"):
            doc_name = filename[: -len("_answers.json")]
        else:
            continue
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                gt_answers = [str(x) if not isinstance(x, str) else x for x in data]
            elif isinstance(data, dict):
                # try sort by numeric keys if any
                try:
                    items = sorted(data.items(), key=lambda kv: int(kv[0]))
                except Exception:
                    items = list(data.items())
                gt_answers = [str(v) if not isinstance(v, str) else v for _, v in items]
            else:
                continue
            mapping[doc_name] = gt_answers
        except Exception:
            continue
    return mapping


def load_baseline_results(results_path: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Load baseline results JSONL.
    Returns: { doc_name: { question_text: { K_str: answer } } }
    """
    store: Dict[str, Dict[str, Dict[str, str]]] = {}
    if not os.path.exists(results_path):
        return store
    with open(results_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            doc = obj.get('Document name')
            q = obj.get('Question')
            k = obj.get('K')  # e.g., "3%"
            ans = obj.get('Answer')
            if not (doc and q and k and (ans is not None)):
                continue
            store.setdefault(doc, {}).setdefault(q, {})[k] = ans
    return store


def assert_baseline_complete(
    baseline: Dict[str, Dict[str, Dict[str, str]]],
    gt: Dict[str, List[str]],
    questions: List[str],
    ks: Tuple[str, ...],
) -> None:
    for doc_name, gt_answers in gt.items():
        base_doc = baseline.get(doc_name)
        if not base_doc:
            raise SystemExit(f"Baseline missing document: {doc_name}")
        n = min(len(gt_answers), len(questions))
        for idx in range(n):
            q_text = questions[idx]
            base_q = base_doc.get(q_text)
            if not base_q:
                raise SystemExit(f"Baseline missing answer for doc={doc_name}, question index={idx}")
            missing_ks = [k for k in ks if k not in base_q]
            if missing_ks:
                raise SystemExit(
                    f"Baseline missing K={missing_ks} for doc={doc_name}, question index={idx}"
                )


def load_completed_detail(detail_path: str) -> Tuple[set, float]:
    """Return (completed_keys, cumulative_cost_usd) from existing detail file.
    completed_keys contains tuples (doc_name, idx, K_str).
    """
    completed = set()
    cumulative_cost = 0.0
    if not os.path.exists(detail_path):
        return completed, 0.0
    with open(detail_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            doc = obj.get("Document name")
            idx = obj.get("QuestionIndex")
            k = obj.get("K")
            jt = obj.get("JudgeTokens", 0)
            if doc is not None and idx is not None and k is not None:
                completed.add((doc, int(idx), k))
            try:
                cumulative_cost += float(jt) * (2.5 / 1_000_000)
            except Exception:
                pass
    return completed, cumulative_cost


def normalize_exact(s: str) -> str:
    return (s or "").strip()


def estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def equal_llm(res1: str, res2: str, question: str) -> Tuple[bool, int]:
    instruction = (
        "I have two answers to the given question. If these two answers are "
        "equivalent in meaning, return True; otherwise, return False. Ignore minor wording differences. "
        "Do not provide any explanation. "
        + "Answer 1: " + str(res1) + " Answer 2: " + str(res2) + " Question: " + str(question)
    )
    tokens = estimate_tokens(instruction)
    try:
        resp = gpt_4o_azure(instruction, max_tokens=8, temperature=0)
        if isinstance(resp, str) and ('true' in resp.lower()):
            return True, tokens
    except Exception:
        pass
    return False, tokens


def evaluate(
    results_path: str = RESULTS_PATH,
    questions_path: str = QUESTIONS_PATH,
    gt_dir: str = GROUND_TRUTH_DIR,
    out_path: str = OUTPUT_SUMMARY_JSON,
    detail_out_path: str = OUTPUT_DETAIL_JSONL,
    ks: Tuple[str, ...] = ("1%", "3%", "5%"),
) -> None:
    questions = load_questions(questions_path)
    if not questions:
        print("No questions loaded.")
        return

    gt = load_ground_truth(gt_dir)
    if not gt:
        print("No ground-truth files found.")
        return

    baseline = load_baseline_results(results_path)
    if not baseline:
        print("No baseline results found.")
        return

    # Ensure baseline completeness (stop program if incomplete)
    assert_baseline_complete(baseline, gt, questions, ks)
 
    # Accumulators per K (will include prior completed detail records)
    totals: Dict[str, int] = {k: 0 for k in ks}
    finals: Dict[str, int] = {k: 0 for k in ks}
    # Resume support: load completed detailed comparisons and initial cost
    completed_pairs, cumulative_cost_usd = load_completed_detail(detail_out_path)

    # Also pre-count totals/finals from existing detail file so accuracies reflect all processed so far
    if os.path.exists(detail_out_path):
        with open(detail_out_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                k = obj.get("K")
                final = obj.get("FinalEquivalent")
                if k in totals:
                    totals[k] += 1
                    if final:
                        finals[k] += 1

    docs = list(gt.items())
    total_docs = len(docs)
    for di, (doc_name, gt_answers) in enumerate(docs):
        # match baseline doc entries by exact document name
        base_doc = baseline.get(doc_name)
        if not base_doc:
            print(f"SKIP: baseline missing for {doc_name}")
            continue

        # decide how many questions to evaluate
        n = min(len(gt_answers), len(questions))

        for idx in range(n):
            q_text = questions[idx]
            gt_ans = gt_answers[idx]
            base_q_entry = base_doc.get(q_text)
            if not base_q_entry:
                # question not present in baseline for this doc
                continue

            for k in ks:
                if k not in base_q_entry:
                    continue
                base_ans = base_q_entry[k]

                # Resume skip if already in detail file
                if (doc_name, idx, k) in completed_pairs:
                    print(f"[RESUME][SKIP] Doc {di+1}/{total_docs} {doc_name} | Q={idx+1}/{n} | K={k}")
                    continue

                totals[k] += 1
                is_exact = normalize_exact(base_ans) == normalize_exact(gt_ans)
                is_final = is_exact
                if not is_exact:
                    eq, judge_tokens = equal_llm(base_ans, gt_ans, q_text)
                    # $2.5 per 1M input tokens
                    cumulative_cost_usd += judge_tokens * (2.5 / 1_000_000)
                    is_final = eq

                if is_final:
                    finals[k] += 1

                # progress
                print(f"[EVAL] Doc {di+1}/{total_docs} {doc_name} | Q={idx+1}/{n} | K={k} | exact={is_exact} | final={is_final}")
                print(f"[COST_SUM] ${cumulative_cost_usd:.6f} so far")

                # persist detailed record for resume
                detail_rec = {
                    "Document name": doc_name,
                    "QuestionIndex": idx,
                    "K": k,
                    "Exact": is_exact,
                    "FinalEquivalent": is_final,
                    "BaselineAnswer": base_ans,
                    "GroundTruth": gt_ans,
                    "Question": q_text,
                    "JudgeTokens": judge_tokens if not is_exact else 0,
                }
                with open(detail_out_path, 'a') as dout:
                    dout.write(json.dumps(detail_rec, ensure_ascii=False) + "\n")
                completed_pairs.add((doc_name, idx, k))

    # Prepare summary accuracies
    summary = {}
    for k in ks:
        acc = (finals[k] / totals[k]) if totals[k] else 0.0
        key = f"acc_{k}"
        summary[key] = acc

    # Write only the three accuracy values
    with open(out_path, 'w') as out:
        out.write(json.dumps(summary, ensure_ascii=False))

    print(summary)


if __name__ == "__main__":
    evaluate()


