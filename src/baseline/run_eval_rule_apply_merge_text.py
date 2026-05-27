"""Apply generated rules with rule_apply_merge and evaluate on text datasets."""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import warnings
from pathlib import Path
from statistics import mean
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from rule_apply.merge import rule_apply_merge

_FINANCE_SPLITS = {
    "sampled": _ROOT / "data/financebench/sample/multi_cluster/random/sample_doc_labels.json",
    "unsampled": _ROOT / "data/financebench/sample/multi_cluster/random/unsampled_doc_labels.json",
}

_DATASET_CONFIG = {
    "financebench": {
        "queries_file": _ROOT / "data/financebench/multi_clsuter_queries.txt",
        "text_dir": _ROOT / "data/financebench/text",
        "labels_file": None,
    },
    "court": {
        "queries_file": _ROOT / "data/court/queries.json",
        "text_dir": _ROOT / "data/court/text",
        "labels_file": _ROOT / "data/court/all_labels.json",
    },
    "nopv": {
        "queries_file": _ROOT / "data/nopv/queries.json",
        "text_dir": _ROOT / "data/nopv/text",
        "labels_file": _ROOT / "data/nopv/all_labels.json",
    },
    "officeqa": {
        "queries_file": _ROOT / "data/officeqa/queries.json",
        "text_dir": _ROOT / "data/officeqa/text",
        "labels_file": _ROOT / "data/officeqa/all_labels.json",
    },
}

_APPLY_SYSTEM_BY_DATASET = {
    "financebench": """\
You are a financial document QA assistant.
You are given a passage extracted from a financial filing and a question.
Answer the question using only the provided passage.
If the passage does not contain enough information to answer, reply exactly "NOT FOUND".
Return only the answer, as a short value or phrase, not a full sentence.""",
    "court": """\
You are a legal document QA assistant.
You are given a passage extracted from a court opinion and a question.
Answer the question using only the provided passage.
If the passage does not contain enough information to answer, reply exactly "NOT FOUND".
Return only the answer, as a short value or phrase, not a full sentence.""",
    "nopv": """\
You are a document QA assistant.
You are given a passage extracted from a regulatory or inspection document and a question.
Answer the question using only the provided passage.
If the passage does not contain enough information to answer, reply exactly "NOT FOUND".
Return only the answer, as a short value or phrase, not a full sentence.""",
    "officeqa": """\
You are a document QA assistant.
You are given a passage extracted from an OfficeQA text document and a question.
Answer the question using only the provided passage.
If the passage does not contain enough information to answer, reply exactly "NOT FOUND".
Return only the answer, as a short value or phrase, not a full sentence.""",
}

_JUDGE_SYSTEM_BY_DATASET = {
    "financebench": """\
You are an answer equivalence judge for a financial document QA system.
Judge whether the predicted answer is semantically equivalent to the ground truth.
Equivalence rules:
- Treat "2017" and "year 2017" as the same
- Treat "$4.5 billion" and "4,500 million" as the same if numerically equal
- Ignore leading/trailing whitespace, punctuation, and capitalization differences
- If the predicted answer is "NOT_FOUND", "NOT FOUND", or null, always judge as incorrect
Reply with exactly one word: CORRECT or INCORRECT""",
    "court": """\
You are an answer equivalence judge for a legal document QA system.
Judge whether the predicted answer is semantically equivalent to the ground truth.
Equivalence rules:
- Ignore capitalization, punctuation, and leading/trailing whitespace differences
- Treat abbreviations and full forms as equivalent (e.g. "9th Cir." and "Ninth Circuit")
- If the predicted answer is "NOT_FOUND", "NOT FOUND", or null, always judge as incorrect
Reply with exactly one word: CORRECT or INCORRECT""",
    "nopv": """\
You are an answer equivalence judge for a document QA system.
Judge whether the predicted answer is semantically equivalent to the ground truth.
Equivalence rules:
- Ignore capitalization, punctuation, and leading/trailing whitespace differences
- If the predicted answer is "NOT_FOUND", "NOT FOUND", or null, always judge as incorrect
Reply with exactly one word: CORRECT or INCORRECT""",
    "officeqa": """\
You are an answer equivalence judge for a document QA system.
Judge whether the predicted answer is semantically equivalent to the ground truth.
Equivalence rules:
- Ignore capitalization, punctuation, and leading/trailing whitespace differences
- If the predicted answer is "NOT_FOUND", "NOT FOUND", or null, always judge as incorrect
Reply with exactly one word: CORRECT or INCORRECT""",
}


class _TimeoutExpired(RuntimeError):
    pass


def _run_with_timeout(timeout_seconds: int | None, fn, *args, **kwargs):
    if not timeout_seconds or timeout_seconds <= 0:
        return fn(*args, **kwargs)

    def _handle_timeout(signum, frame):
        raise _TimeoutExpired(f"timed out after {timeout_seconds}s")

    prev_handler = signal.getsignal(signal.SIGALRM)
    prev_timer = signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    signal.signal(signal.SIGALRM, _handle_timeout)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev_handler)
        if prev_timer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, *prev_timer)


def _make_slug(text: str) -> str:
    s = text.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60]


def _count_tokens(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(0, int(len(text) / 4))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _split_pages(text: str) -> list[str]:
    parts = re.split(r"\f+", text)
    return parts if parts else [text]


def _load_txt_doc(doc_name: str, doc_path: Path) -> dict[str, Any]:
    raw = _read_text(doc_path)
    pages_raw = _split_pages(raw)

    pages: list[dict[str, Any]] = []
    paragraphs: list[dict[str, Any]] = []
    lines: list[dict[str, Any]] = []

    global_line_idx = 0
    global_para_idx = 0

    for page_idx, page_text in enumerate(pages_raw, start=1):
        pages.append({"page_no": page_idx, "text": page_text})

        page_lines = page_text.splitlines()
        for line_no, line in enumerate(page_lines, start=1):
            lines.append(
                {
                    "page_no": page_idx,
                    "line_no": line_no,
                    "line_index": global_line_idx,
                    "text": line,
                }
            )
            global_line_idx += 1

        para_no = 0
        current: list[str] = []
        for line in page_lines + [""]:
            if line.strip():
                current.append(line)
                continue
            if current:
                para_no += 1
                paragraphs.append(
                    {
                        "page_no": page_idx,
                        "paragraph_no": para_no,
                        "paragraph_index": global_para_idx,
                        "text": "\n".join(current).strip(),
                    }
                )
                global_para_idx += 1
                current = []

    return {
        "doc_name": doc_name,
        "doc_path": str(doc_path.resolve()),
        "text": raw,
        "pages": pages,
        "paragraphs": paragraphs,
        "lines": lines,
    }


def _load_questions(dataset: str) -> list[str]:
    path = _DATASET_CONFIG[dataset]["queries_file"]
    if path.suffix == ".txt":
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    return [item["text"] for item in data]


def _load_labels(dataset: str, split: str) -> dict[str, dict]:
    if dataset != "financebench":
        path = _DATASET_CONFIG[dataset]["labels_file"]
        return json.loads(path.read_text(encoding="utf-8"))

    if split != "all_docs":
        path = _FINANCE_SPLITS[split]
        return json.loads(path.read_text(encoding="utf-8"))

    merged: dict[str, dict] = {}
    for path in (_FINANCE_SPLITS["sampled"], _FINANCE_SPLITS["unsampled"]):
        data = json.loads(path.read_text(encoding="utf-8"))
        for pdf_key, doc_labels in data.items():
            if pdf_key in merged and merged[pdf_key] != doc_labels:
                raise ValueError(f"Conflicting labels for duplicated doc: {pdf_key}")
            merged[pdf_key] = doc_labels
    return merged


def _select_docs(
    labels: dict[str, dict],
    *,
    start_doc: int,
    max_docs: int | None,
    latest_docs: bool,
) -> dict[str, dict]:
    items = sorted(labels.items())
    if latest_docs:
        if max_docs is None:
            raise SystemExit("--latest-docs requires --max-docs")
        items = items[-max_docs:]
    else:
        if start_doc:
            items = items[start_doc:]
        if max_docs is not None:
            items = items[:max_docs]
    return dict(items)


def _judge(dataset: str, question: str, ground_truth, predicted, gpt54_mod) -> bool:
    if ground_truth is None or predicted is None:
        return False
    gt_str = json.dumps(ground_truth) if not isinstance(ground_truth, str) else ground_truth
    pred_str = str(predicted)
    resp = gpt54_mod.client.chat.completions.create(
        model=gpt54_mod.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM_BY_DATASET[dataset]},
            {"role": "user", "content": f"Question: {question}\nGround Truth: {gt_str}\nPredicted: {pred_str}"},
        ],
        max_completion_tokens=10,
        temperature=0.0,
    )
    verdict = (resp.choices[0].message.content or "").strip().lower()
    if verdict not in ("correct", "incorrect"):
        warnings.warn(f"Unexpected judge verdict: {verdict!r}")
    return verdict == "correct"


def _find_rule_reports(results_root: Path, question_slug_prefix: str | None) -> list[tuple[dict[str, Any], Path]]:
    report_paths = sorted(results_root.glob("*_rule_gen.json"))
    if not report_paths:
        raise FileNotFoundError(f"No rule generation reports found in {results_root}")

    matches: list[tuple[dict[str, Any], Path]] = []
    for path in report_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        slug = data.get("question_slug", "")
        if question_slug_prefix and not slug.startswith(question_slug_prefix):
            continue
        matches.append((data, path))
    if not matches:
        raise FileNotFoundError(
            f"No rule generation report in {results_root} matches question slug prefix {question_slug_prefix!r}"
        )
    return matches


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply generated rules with rule_apply_merge and evaluate on text datasets")
    ap.add_argument("--dataset", choices=sorted(_DATASET_CONFIG), required=True)
    ap.add_argument("--rule-gen-output-name", required=True)
    ap.add_argument("--question-slug", default=None, help="Optional slug prefix to restrict applied questions")
    ap.add_argument("--model", default="gpt54", help="Answer-generation model passed into rule_apply_merge")
    ap.add_argument("--split", default="all_docs", help="Logical split label for labels loading (default: all_docs)")
    ap.add_argument("--split-name", default=None, help="Optional output split label override")
    ap.add_argument("--start-doc", type=int, default=0)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--latest-docs", action="store_true")
    ap.add_argument("--output-name", default=None, help="Optional output name under results/<dataset>/")
    ap.add_argument("--apply-timeout-seconds", type=int, default=180)
    ap.add_argument("--judge-timeout-seconds", type=int, default=60)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    gpt54_mod = __import__("models.gpt54", fromlist=["client"])

    split_name = args.split_name or args.split
    labels_all = _load_labels(args.dataset, args.split)
    labels_selected = _select_docs(
        labels_all,
        start_doc=args.start_doc,
        max_docs=args.max_docs,
        latest_docs=args.latest_docs,
    )

    text_dir = _DATASET_CONFIG[args.dataset]["text_dir"]
    selected_docs: dict[str, dict] = {}
    for pdf_key, doc_labels in sorted(labels_selected.items()):
        doc_name = pdf_key.replace(".pdf", "")
        doc_path = text_dir / f"{doc_name}.txt"
        if not doc_path.exists():
            print(f"SKIP (no txt): {doc_name}")
            continue
        selected_docs[doc_name] = {
            "labels": doc_labels,
            "path": doc_path,
        }

    rules_root = _ROOT / "rules" / args.dataset / args.rule_gen_output_name
    results_root = _ROOT / "results" / args.dataset / args.rule_gen_output_name
    report_entries = _find_rule_reports(results_root, args.question_slug)

    out_name = args.output_name or f"{args.rule_gen_output_name}/rule_apply_merge"
    out_base = _ROOT / "results" / args.dataset / out_name
    out_base.mkdir(parents=True, exist_ok=True)
    trace_dir = out_base / "_trace"
    trace_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "dataset": args.dataset,
        "strategy": "rule_apply_merge",
        "split": split_name,
        "rule_gen_output_name": args.rule_gen_output_name,
        "answer_model": args.model,
        "apply_timeout_seconds": args.apply_timeout_seconds,
        "judge_timeout_seconds": args.judge_timeout_seconds,
        "selected_docs": sorted(selected_docs),
        "doc_count": len(selected_docs),
        "question_count": len(report_entries),
        "question_slug_prefix": args.question_slug,
    }
    (out_base / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"dataset={args.dataset}  answer_model={args.model}  "
        f"questions={len(report_entries)}  docs={len(selected_docs)}"
    )
    print(f"rule_gen_output={args.rule_gen_output_name}")
    print(f"output={out_base}\n")

    all_summaries: list[dict[str, Any]] = []
    for report, report_path in report_entries:
        question = report["question"]
        question_slug = report["question_slug"]
        rule_names = list(report.get("rules_final") or [])
        if not rule_names:
            print(f"SKIP {question_slug} (no rules_final)")
            continue

        rule_question_dir = None
        for path in sorted(rules_root.glob(f"*_{question_slug}")):
            if path.is_dir():
                rule_question_dir = path.name
                break
        if rule_question_dir is None:
            print(f"SKIP {question_slug} (no rule directory)")
            continue

        print(f"[{question_slug}] {question}")
        q_dir = out_base / question_slug
        q_dir.mkdir(parents=True, exist_ok=True)
        per_doc_results: list[dict[str, Any]] = []

        for doc_name in sorted(selected_docs):
            out_file = q_dir / f"{doc_name}.json"
            if args.skip_existing and out_file.exists():
                try:
                    per_doc_results.append(json.loads(out_file.read_text(encoding="utf-8")))
                    continue
                except Exception:
                    pass

            doc_info = selected_docs[doc_name]
            document = _load_txt_doc(doc_name, doc_info["path"])
            ground_truth = doc_info["labels"].get(question)
            doc_token_count = _count_tokens(document["text"])

            try:
                apply_result = _run_with_timeout(
                    args.apply_timeout_seconds,
                    rule_apply_merge,
                    document=document,
                    rule_names=rule_names,
                    question_slug=rule_question_dir,
                    question=question,
                    model_name=args.model,
                    rules_dir=str(rules_root),
                    output_dir=str(trace_dir),
                    system_prompt=_APPLY_SYSTEM_BY_DATASET[args.dataset],
                )
                answer = apply_result.get("predicted_answer")
                status = "ok"
            except _TimeoutExpired as exc:
                print(f"  TIMEOUT {doc_name}: {exc}")
                apply_result = {
                    "predicted_answer": None,
                    "latency_seconds": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "retrieved_token_count": 0,
                    "rules_with_hits": [],
                    "rules_with_no_hits": list(rule_names),
                    "num_spans_before_dedup": 0,
                    "num_spans_after_dedup": 0,
                    "retrieved_spans": [],
                    "retrieved_text": "",
                }
                answer = None
                status = "apply_timeout"
            except Exception as exc:
                print(f"  ERROR {doc_name}: {exc}")
                apply_result = {
                    "predicted_answer": None,
                    "latency_seconds": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "retrieved_token_count": 0,
                    "rules_with_hits": [],
                    "rules_with_no_hits": list(rule_names),
                    "num_spans_before_dedup": 0,
                    "num_spans_after_dedup": 0,
                    "retrieved_spans": [],
                    "retrieved_text": "",
                }
                answer = None
                status = "error"

            try:
                correct = _run_with_timeout(
                    args.judge_timeout_seconds,
                    _judge,
                    args.dataset,
                    question,
                    ground_truth,
                    answer,
                    gpt54_mod,
                )
            except _TimeoutExpired as exc:
                print(f"  JUDGE_TIMEOUT {doc_name}: {exc}")
                correct = False
                if status == "ok":
                    status = "judge_timeout"
            cost_ratio = round(apply_result.get("input_tokens", 0) / max(doc_token_count, 1), 4)
            rule_set_slug = "__".join(sorted(rule_names))[:120]
            record = {
                "doc_name": doc_name,
                "question": question,
                "question_slug": question_slug,
                "split": split_name,
                "ground_truth": ground_truth,
                "answer": answer,
                "correct": correct,
                "status": status,
                "input_tokens": apply_result.get("input_tokens", 0),
                "output_tokens": apply_result.get("output_tokens", 0),
                "latency_seconds": apply_result.get("latency_seconds", 0.0),
                "model": args.model,
                "doc_token_count": doc_token_count,
                "cost_ratio": cost_ratio,
                "retrieved_token_count": apply_result.get("retrieved_token_count", 0),
                "rules_with_hits": apply_result.get("rules_with_hits", []),
                "rules_with_no_hits": apply_result.get("rules_with_no_hits", []),
                "num_spans_before_dedup": apply_result.get("num_spans_before_dedup", 0),
                "num_spans_after_dedup": apply_result.get("num_spans_after_dedup", 0),
                "rule_names": rule_names,
                "rule_question_dir": rule_question_dir,
                "rule_generation_report_path": str(report_path.relative_to(_ROOT)),
                "rule_apply_trace_path": str(
                    (trace_dir / rule_question_dir / f"{rule_set_slug}_merge.json").relative_to(_ROOT)
                ),
            }
            out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            per_doc_results.append(record)
            marker = "✓" if correct else "✗"
            print(
                f"  {marker} {doc_name:<55} {answer!r}  "
                f"in={record['input_tokens']}  cost={record['cost_ratio']:.4f}  lat={record['latency_seconds']:.1f}s"
            )

        if not per_doc_results:
            continue

        n = len(per_doc_results)
        n_correct = sum(r["correct"] for r in per_doc_results)
        summary = {
            "question": question,
            "question_slug": question_slug,
            "split": split_name,
            "model": args.model,
            "n": n,
            "n_correct": n_correct,
            "accuracy": round(n_correct / n, 4),
            "avg_input_tokens": round(mean(r["input_tokens"] for r in per_doc_results), 1),
            "avg_output_tokens": round(mean(r["output_tokens"] for r in per_doc_results), 1),
            "avg_latency_seconds": round(mean(r["latency_seconds"] for r in per_doc_results), 2),
            "avg_retrieved_tokens": round(mean(r["retrieved_token_count"] for r in per_doc_results), 1),
            "avg_cost_ratio": round(mean(r["cost_ratio"] for r in per_doc_results), 4),
            "rule_gen_output_name": args.rule_gen_output_name,
            "rule_question_dir": rule_question_dir,
            "rule_names": rule_names,
        }
        all_summaries.append(summary)
        print(
            f"\nQuestion: {question[:70]}\n"
            f"  accuracy={summary['accuracy']:.4f} ({n_correct}/{n})  "
            f"avg_cost={summary['avg_cost_ratio']:.4f}  avg_lat={summary['avg_latency_seconds']:.1f}s\n"
        )

    (out_base / "summary.json").write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if all_summaries:
        print(f"\n{'=' * 60}")
        print(f"  mean accuracy : {mean(s['accuracy'] for s in all_summaries):.4f}")
        print(f"  mean cost     : {mean(s['avg_cost_ratio'] for s in all_summaries):.4f}")
        print(f"  mean latency  : {mean(s['avg_latency_seconds'] for s in all_summaries):.1f}s")
        print(f"  results in    : {out_base}/")


if __name__ == "__main__":
    main()
