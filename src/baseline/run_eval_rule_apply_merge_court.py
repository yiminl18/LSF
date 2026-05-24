"""Apply generated court rules with rule_apply_merge and evaluate on all docs."""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from statistics import mean
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from rule_apply_merge import rule_apply_merge

DATASET = "court"
QUERIES_FILE = _ROOT / "data/court/queries.json"
TEXT_DIR = _ROOT / "data/court/text"
LABELS_FILE = _ROOT / "data/court/all_labels.json"

_COURT_APPLY_SYSTEM = """\
You are a legal document QA assistant.
You are given a passage extracted from a court opinion and a question.
Answer the question using only the provided passage.
If the passage does not contain enough information to answer, reply exactly "NOT FOUND".
Return only the answer, as a short value or phrase, not a full sentence."""

_JUDGE_SYSTEM = """\
You are an answer equivalence judge for a legal document QA system.
Judge whether the predicted answer is semantically equivalent to the ground truth.
Equivalence rules:
- Ignore capitalization, punctuation, and leading/trailing whitespace differences
- Treat abbreviations and full forms as equivalent (e.g. "9th Cir." and "Ninth Circuit")
- If the predicted answer is "NOT_FOUND", "NOT FOUND", or null, always judge as incorrect
Reply with exactly one word: CORRECT or INCORRECT"""


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


def _judge(question: str, ground_truth, predicted, gpt54_mod) -> bool:
    if ground_truth is None or predicted is None:
        return False
    gt_str = json.dumps(ground_truth) if not isinstance(ground_truth, str) else ground_truth
    pred_str = str(predicted)
    resp = gpt54_mod.client.chat.completions.create(
        model=gpt54_mod.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": f"Question: {question}\nGround Truth: {gt_str}\nPredicted: {pred_str}"},
        ],
        max_completion_tokens=10,
        temperature=0.0,
    )
    verdict = (resp.choices[0].message.content or "").strip().lower()
    if verdict not in ("correct", "incorrect"):
        warnings.warn(f"Unexpected judge verdict: {verdict!r}")
    return verdict == "correct"


def _find_rule_report(results_root: Path, question_slug_prefix: str | None) -> tuple[dict[str, Any], Path]:
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
    if len(matches) > 1:
        raise ValueError(
            f"Multiple rule generation reports matched {question_slug_prefix!r}; be more specific"
        )
    return matches[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply generated court rules with rule_apply_merge and evaluate them")
    ap.add_argument(
        "--rule-gen-output-name",
        required=True,
        help="Output name previously used by run_eval_rule_full_data.py under rules/court/ and results/court/rule_gen/",
    )
    ap.add_argument(
        "--question-slug",
        default=None,
        help="Question slug prefix to evaluate. If omitted, uses the only rule-gen report in the output root.",
    )
    ap.add_argument(
        "--model",
        default="gpt54",
        help="Model name passed into rule_apply_merge for answer generation (default: gpt54)",
    )
    ap.add_argument(
        "--output-name",
        default=None,
        help="Optional custom output directory name under baseline_results/court/ (default: rule_apply_merge/<rule-gen-output-name>)",
    )
    ap.add_argument("--split-name", default="all_docs")
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    gpt54_mod = __import__("models.gpt54", fromlist=["client"])

    queries_raw = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
    questions_by_slug = {_make_slug(item["text"]): item["text"] for item in queries_raw}
    labels: dict[str, dict] = json.loads(LABELS_FILE.read_text(encoding="utf-8"))

    rules_root = _ROOT / "rules" / DATASET / args.rule_gen_output_name
    results_root = _ROOT / "results" / DATASET / "rule_gen" / args.rule_gen_output_name
    report, report_path = _find_rule_report(results_root, args.question_slug)

    question = report["question"]
    question_slug = report["question_slug"]
    question_text = questions_by_slug.get(question_slug, question)
    rule_names = list(report.get("rules_final") or [])
    if not rule_names:
        raise ValueError(f"Rule report {report_path} has no rules_final")

    rule_question_dir = None
    for path in sorted((rules_root).glob(f"*_{question_slug}")):
        if path.is_dir():
            rule_question_dir = path.name
            break
    if rule_question_dir is None:
        raise FileNotFoundError(f"No rule directory under {rules_root} matches *_{question_slug}")

    out_name = args.output_name or f"rule_apply_merge/{args.rule_gen_output_name}"
    out_base = _ROOT / "baseline_results" / DATASET / out_name
    out_base.mkdir(parents=True, exist_ok=True)
    q_dir = out_base / question_slug
    q_dir.mkdir(parents=True, exist_ok=True)

    merge_trace_dir = _ROOT / "results" / DATASET / "rule_apply_merge" / args.rule_gen_output_name
    merge_trace_dir.mkdir(parents=True, exist_ok=True)

    selected_items = sorted(labels.items())
    if args.max_docs is not None:
        selected_items = selected_items[: args.max_docs]

    metadata = {
        "dataset": DATASET,
        "strategy": "rule_apply_merge",
        "split": args.split_name,
        "rule_gen_output_name": args.rule_gen_output_name,
        "rule_generation_report_path": str(report_path.relative_to(_ROOT)),
        "rule_question_dir": rule_question_dir,
        "question": question_text,
        "question_slug": question_slug,
        "rule_names": rule_names,
        "answer_model": args.model,
        "selected_docs": [pdf_key.replace(".pdf", "") for pdf_key, _ in selected_items],
        "doc_count": len(selected_items),
    }
    (out_base / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"rule_gen_output={args.rule_gen_output_name}")
    print(f"question={question_text}")
    print(f"rules={len(rule_names)}")
    print(f"docs={len(selected_items)}")
    print(f"output={out_base}\n")

    per_doc_results: list[dict[str, Any]] = []
    for pdf_key, doc_labels in selected_items:
        doc_name = pdf_key.replace(".pdf", "")
        out_file = q_dir / f"{doc_name}.json"

        if args.skip_existing and out_file.exists():
            try:
                per_doc_results.append(json.loads(out_file.read_text(encoding="utf-8")))
                continue
            except Exception:
                pass

        doc_path = TEXT_DIR / f"{doc_name}.txt"
        if not doc_path.exists():
            print(f"  SKIP (no txt): {doc_name}")
            continue

        document = _load_txt_doc(doc_name, doc_path)
        ground_truth = doc_labels.get(question_text)
        doc_token_count = _count_tokens(document["text"])

        try:
            apply_result = rule_apply_merge(
                document=document,
                rule_names=rule_names,
                question_slug=rule_question_dir,
                question=question_text,
                model_name=args.model,
                rules_dir=str(rules_root),
                output_dir=str(merge_trace_dir),
                system_prompt=_COURT_APPLY_SYSTEM,
            )
            answer = apply_result.get("predicted_answer")
            status = "ok"
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

        correct = _judge(question_text, ground_truth, answer, gpt54_mod)
        cost_ratio = round(apply_result.get("input_tokens", 0) / max(doc_token_count, 1), 4)
        rule_set_slug = "__".join(sorted(rule_names))[:120]

        record = {
            "doc_name": doc_name,
            "question": question_text,
            "question_slug": question_slug,
            "split": args.split_name,
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
                (merge_trace_dir / rule_question_dir / f"{rule_set_slug}_merge.json").relative_to(_ROOT)
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
        return

    n = len(per_doc_results)
    n_correct = sum(r["correct"] for r in per_doc_results)
    summary = [
        {
            "question": question_text,
            "question_slug": question_slug,
            "split": args.split_name,
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
    ]
    (out_base / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n{'=' * 60}")
    print(f"  accuracy     : {summary[0]['accuracy']:.4f}")
    print(f"  avg cost     : {summary[0]['avg_cost_ratio']:.4f}")
    print(f"  avg latency  : {summary[0]['avg_latency_seconds']:.1f}s")
    print(f"  results in   : {out_base}/")


if __name__ == "__main__":
    main()
