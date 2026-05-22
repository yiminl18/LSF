"""Run agentic Codex QA baseline on the court dataset.

Output layout:
    baseline_results/court/agentic_codex_qa_<model>/all_docs/
        <question_slug>/<doc_name>.json
        summary.json

Usage:
    python src/baseline/run_eval_court.py --model gpt54
    python src/baseline/run_eval_court.py --model gpt54mini
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from statistics import mean

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

DATASET       = "court"
QUERIES_FILE  = _ROOT / "data/court/queries.json"
TEXT_DIR      = _ROOT / "data/court/text"
LABELS_FILE   = _ROOT / "data/court/all_labels.json"

_MODEL_ALIASES: dict[str, str] = {
    "gpt54":     "gpt-5.4",
    "gpt54mini": "gpt-5.4-mini",
}

_AGENT_PROMPT_TEMPLATE = """\
Answer the following question about the document at the path below.

QUESTION : {question}
DOCUMENT : {doc_path}

You are running as a Codex agent. Use your default tools to read DOCUMENT.
It is a plain-text file containing the full text of a court filing.

Steps:
1. Read and search the document as needed.
2. Locate the shortest document-supported answer to QUESTION.
3. Output exactly one line:
   AGENTIC_QA_DONE answer=<your answer>

Rules:
- Give a short, direct answer (a name, date, number, etc.) with no explanation.
- If the answer is not found in the document, output:
  AGENTIC_QA_DONE answer=NOT_FOUND
- Do not edit files.
- Do not output anything else after the AGENTIC_QA_DONE line.
"""

_JUDGE_SYSTEM = """\
You are an answer equivalence judge for a legal document QA system.
Judge whether the predicted answer is semantically equivalent to the ground truth.
Equivalence rules:
- Ignore capitalization, punctuation, and leading/trailing whitespace differences
- Treat abbreviations and full forms as equivalent (e.g. "9th Cir." and "Ninth Circuit")
- If the predicted answer is "NOT_FOUND", "NOT FOUND", or null, always judge as incorrect
Reply with exactly one word: CORRECT or INCORRECT"""


def _make_slug(q: str) -> str:
    s = q.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60]


def _parse_answer(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("AGENTIC_QA_DONE"):
            rest = line[len("AGENTIC_QA_DONE"):].strip()
            if rest.startswith("answer="):
                return rest[len("answer="):].strip()
    return None


def _parse_codex_events(jsonl_text: str) -> dict:
    usage: dict[str, int] = {}
    thread_id = None
    error_message = None
    event_count = 0
    for raw_line in jsonl_text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        event_count += 1
        if event.get("type") == "thread.started":
            thread_id = event.get("thread_id")
        elif event.get("type") == "turn.completed":
            usage = event.get("usage") or {}
        elif event.get("type") in {"error", "turn.failed"}:
            error_message = event.get("message") or str(event.get("error") or "")
    return {"usage": usage, "thread_id": thread_id, "error_message": error_message, "event_count": event_count}


def _run_codex(doc_path: Path, question: str, model: str, timeout: int, log_dir: Path | None, log_stem: str | None) -> dict:
    codex_bin = shutil.which("codex")
    resolved_model = _MODEL_ALIASES.get(model, model)
    if not codex_bin:
        return {"status": "error", "answer": None, "input_tokens": 0, "output_tokens": 0,
                "latency_seconds": 0.0, "model": resolved_model, "error_message": "codex CLI not found"}

    prompt = _AGENT_PROMPT_TEMPLATE.format(question=question, doc_path=str(doc_path.resolve()))

    log_path = last_message_path = None
    if log_dir and log_stem:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{log_stem}.codex.jsonl"
        last_message_path = log_dir / f"{log_stem}.codex.last.txt"

    with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
        tmp_last_path = Path(tmp.name)

    cmd = [
        codex_bin, "--ask-for-approval", "never", "exec",
        "--json", "--color", "never", "--model", resolved_model,
        "--cd", str(_ROOT), "--sandbox", "danger-full-access",
        "--output-last-message", str(tmp_last_path), prompt,
    ]

    t0 = time.time()
    try:
        res = subprocess.run(cmd, input="", capture_output=True, text=True, cwd=str(_ROOT), timeout=timeout)
    except subprocess.TimeoutExpired:
        tmp_last_path.unlink(missing_ok=True)
        return {"status": "timeout", "answer": None, "input_tokens": 0, "output_tokens": 0,
                "latency_seconds": round(time.time() - t0, 2), "model": resolved_model}

    latency = round(time.time() - t0, 2)
    jsonl_text = res.stdout or ""
    last_text = ""
    try:
        last_text = tmp_last_path.read_text(encoding="utf-8", errors="replace")
    finally:
        tmp_last_path.unlink(missing_ok=True)

    if log_path:
        log_path.write_text(jsonl_text, encoding="utf-8")
    if last_message_path:
        last_message_path.write_text(last_text, encoding="utf-8")

    parsed = _parse_codex_events(jsonl_text)
    usage = parsed["usage"]
    answer = _parse_answer(last_text) or _parse_answer(jsonl_text)

    cached = int(usage.get("cached_input_tokens") or 0)
    input_tokens = int(usage.get("input_tokens") or 0) + cached
    output_tokens = int(usage.get("output_tokens") or 0)

    return {
        "status": "ok" if res.returncode == 0 else f"exit_{res.returncode}",
        "answer": answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_seconds": latency,
        "model": resolved_model,
        "cached_input_tokens": cached,
        "reasoning_output_tokens": int(usage.get("reasoning_output_tokens") or 0),
        "codex_thread_id": parsed["thread_id"],
        "codex_event_count": parsed["event_count"],
        "codex_error_message": parsed["error_message"],
        "codex_log_path": str(log_path.relative_to(_ROOT)) if log_path else None,
        "codex_last_message_path": str(last_message_path.relative_to(_ROOT)) if last_message_path else None,
        "stderr": (res.stderr or "")[:500],
    }


def _judge(question: str, ground_truth, predicted, gpt54_mod) -> bool:
    if ground_truth is None or predicted is None:
        return False
    gt_str   = json.dumps(ground_truth) if not isinstance(ground_truth, str) else ground_truth
    pred_str = str(predicted)
    resp = gpt54_mod.client.chat.completions.create(
        model=gpt54_mod.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user",   "content": f"Question: {question}\nGround Truth: {gt_str}\nPredicted: {pred_str}"},
        ],
        max_completion_tokens=10,
        temperature=0.0,
    )
    verdict = (resp.choices[0].message.content or "").strip().lower()
    if verdict not in ("correct", "incorrect"):
        warnings.warn(f"Unexpected judge verdict: {verdict!r}")
    return verdict == "correct"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt54", help="gpt54 or gpt54mini")
    ap.add_argument("--question-slug", default=None)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    gpt54_mod = importlib.import_module("models.gpt54")

    queries_raw = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
    questions = [q["text"] for q in queries_raw]
    labels: dict = json.loads(LABELS_FILE.read_text(encoding="utf-8"))

    out_base = _ROOT / "baseline_results" / DATASET / f"agentic_codex_qa_{args.model}" / "all_docs"
    out_base.mkdir(parents=True, exist_ok=True)

    if args.max_docs:
        items = sorted(labels.items())[:args.max_docs]
        labels = dict(items)

    print(f"model={args.model}  questions={len(questions)}  docs={len(labels)}  output={out_base}")
    print()

    all_summaries: list[dict] = []

    for question in questions:
        slug = _make_slug(question)
        if args.question_slug and not slug.startswith(args.question_slug):
            continue

        q_dir = out_base / slug
        q_dir.mkdir(exist_ok=True)
        per_doc_results: list[dict] = []

        for pdf_key, doc_labels in sorted(labels.items()):
            doc_name = pdf_key.replace(".pdf", "")
            doc_path = TEXT_DIR / f"{doc_name}.txt"
            out_file = q_dir / f"{doc_name}.json"

            if args.skip_existing and out_file.exists():
                try:
                    per_doc_results.append(json.loads(out_file.read_text(encoding="utf-8")))
                except Exception:
                    pass
                continue

            if not doc_path.exists():
                print(f"  SKIP (no txt): {doc_name}")
                continue

            ground_truth = doc_labels.get(question)

            try:
                result = _run_codex(doc_path, question, args.model, args.timeout,
                                    log_dir=q_dir / "logs", log_stem=doc_name)
            except Exception as e:
                print(f"  ERROR {doc_name}: {e}")
                result = {"status": "error", "answer": None, "input_tokens": 0,
                          "output_tokens": 0, "latency_seconds": 0.0, "model": args.model}

            correct = _judge(question, ground_truth, result.get("answer"), gpt54_mod)

            record = {
                "doc_name":        doc_name,
                "question":        question,
                "question_slug":   slug,
                "ground_truth":    ground_truth,
                "answer":          result.get("answer"),
                "correct":         correct,
                "status":          result.get("status"),
                "input_tokens":    result.get("input_tokens", 0),
                "output_tokens":   result.get("output_tokens", 0),
                "latency_seconds": result.get("latency_seconds", 0.0),
                "model":           result.get("model", args.model),
            }
            for k, v in result.items():
                if k not in record and k != "answer":
                    record[k] = v

            out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            per_doc_results.append(record)
            status = "✓" if correct else "✗"
            print(f"  {status} {doc_name}: {record['answer']!r}  in={record['input_tokens']}  lat={record['latency_seconds']:.1f}s")

        if not per_doc_results:
            continue

        n         = len(per_doc_results)
        n_correct = sum(r["correct"] for r in per_doc_results)
        accuracy  = round(n_correct / n, 4)
        avg_in    = round(mean(r["input_tokens"]    for r in per_doc_results), 1)
        avg_out   = round(mean(r["output_tokens"]   for r in per_doc_results), 1)
        avg_lat   = round(mean(r["latency_seconds"] for r in per_doc_results), 2)

        q_summary = {
            "question": question, "question_slug": slug, "model": args.model,
            "n": n, "n_correct": n_correct, "accuracy": accuracy,
            "avg_input_tokens": avg_in, "avg_output_tokens": avg_out, "avg_latency_seconds": avg_lat,
        }
        all_summaries.append(q_summary)
        print(f"\nQuestion: {question[:70]}")
        print(f"  accuracy={accuracy:.2f} ({n_correct}/{n})  avg_in={avg_in:.0f}  avg_lat={avg_lat:.1f}s\n")

    summary_path = out_base / "summary.json"
    existing: list[dict] = []
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    merged = {e["question_slug"]: e for e in existing}
    for s in all_summaries:
        merged[s["question_slug"]] = s
    summary_path.write_text(json.dumps(list(merged.values()), indent=2, ensure_ascii=False), encoding="utf-8")

    if all_summaries:
        print(f"\n{'='*60}")
        print(f"  mean accuracy : {mean(s['accuracy'] for s in all_summaries):.4f}")
        print(f"  mean in-tokens: {mean(s['avg_input_tokens'] for s in all_summaries):.0f}")
        print(f"  mean latency  : {mean(s['avg_latency_seconds'] for s in all_summaries):.1f}s")
        print(f"  results in    : {out_base}/")


if __name__ == "__main__":
    main()
