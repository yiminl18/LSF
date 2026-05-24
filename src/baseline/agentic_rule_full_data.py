"""Strategy 4 rule generation: Agentic Rule Full Data.

This module only handles the rule-generation half. One Codex agent session runs
per question, inspects the plain-text corpus, writes Python retrieval rules,
verifies them against a working sample, and exits with a final rule set plus
rule-generation metadata.

The downstream rule-application/evaluation phase is intentionally separate and
should use existing application code elsewhere in the repo.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_MODEL_ALIASES: dict[str, str] = {
    "gpt54": "gpt-5.4",
    "gpt54mini": "gpt-5.4-mini",
    "gpt-5.4": "gpt-5.4",
    "gpt-5.4-mini": "gpt-5.4-mini",
}

_VERIFY_BUDGET = 30

_QA_SYSTEM = (
    "You are a document QA assistant.\n"
    "Answer the question using only the provided passage.\n"
    "The passage may be a terse extracted snippet that contains only the answer value.\n"
    "If the passage contains the answer, return the answer value directly even when it is not a full sentence.\n"
    "Strip obvious boilerplate labels such as No., Nos., D.C. No., Telephone:, and Filed: when returning the answer.\n"
    'If the passage does not contain enough information, reply exactly "NOT_FOUND".\n'
    "Return only the answer, not an explanation."
)

_JUDGE_SYSTEM_GENERIC = """\
You are an answer equivalence judge for a document QA system.
Judge whether the predicted answer is semantically equivalent to the ground truth.
Equivalence rules:
- Ignore capitalization, punctuation, and leading/trailing whitespace differences
- Treat minor formatting differences in numbers as equivalent
- If the predicted answer is "NOT_FOUND", "NOT FOUND", or null, always judge as incorrect
Reply with exactly one word: CORRECT or INCORRECT"""

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
    "nopv": _JUDGE_SYSTEM_GENERIC,
    "officeqa": _JUDGE_SYSTEM_GENERIC,
}

_AGENT_PROMPT_TEMPLATE = """\
Generate Python retrieval rules for one question over a plain-text corpus.

QUESTION     : {question}
QUESTION_SLUG: {question_slug}
MANIFEST     : {manifest_path}
RULES_DIR    : {rules_dir}
REPORT_PATH  : {report_path}
LEDGER_PATH  : {ledger_path}

You are running as a Codex agent in this repository with default tools.
Use shell/file tools freely, but keep all writes inside RULES_DIR and REPORT_PATH.

Available helper commands:
  python src/baseline/agentic_rule_full_data.py list-docs --manifest {manifest_path}
  python src/baseline/agentic_rule_full_data.py read-doc-txt --manifest {manifest_path} --doc-name <DOC_NAME>
  python src/baseline/agentic_rule_full_data.py compute-cost --manifest {manifest_path}
  python src/baseline/agentic_rule_full_data.py compute-cost --manifest {manifest_path} --doc-names <DOC1> <DOC2>
  python src/baseline/agentic_rule_full_data.py inspect-rule --manifest {manifest_path} --rule-name <RULE_NAME>
  python src/baseline/agentic_rule_full_data.py verify-accuracy --manifest {manifest_path} --doc-names <DOC1> <DOC2> ...

Important:
- `verify-accuracy` is the paid verification tool. Budget: {verify_budget} calls.
- The helper CLI tracks verification token usage in LEDGER_PATH. Use it instead of ad hoc scripts.
- Source documents are plain-text `.txt` files. Do not use reconstructed JSON files.

Rule contract:
- write one file per rule under RULES_DIR
- each file must define exactly one function named `rule_<name>`
- function signature: `def rule_<name>(doc: dict) -> list[dict]:`
- `doc` has:
  - `doc["doc_name"]`
  - `doc["text"]`          full plain text
  - `doc["pages"]`         list[{{page_no, text}}]
  - `doc["paragraphs"]`    list[{{page_no, paragraph_no, text}}]
  - `doc["lines"]`         list[{{page_no, line_no, text}}]
- return a list of span dicts with at least a `text` field; keep page/line metadata when available
- wrap the full body in `try/except Exception: return []`

Recommended workflow:
1. Use `list-docs` and `read-doc-txt` to inspect a small seed sample.
2. Write 1-3 broad rules first.
3. Use `compute-cost` and `verify-accuracy` on your working sample.
4. Expand the working sample only when needed to diagnose failures.
5. Stop when the rules look stable or when more tightening hurts coverage.

Before finishing:
- ensure the final rule set is stored in RULES_DIR
- write REPORT_PATH as JSON with this schema:
  {{
    "working_sample": ["DOC1", "DOC2"],
    "num_working_sample_iterations": 2,
    "rules_final": ["rule_x", "rule_y"],
    "num_rules_rejected": 0,
    "match_rate_sample": 0.95,
    "avg_cost_ratio_sample": 0.01,
    "notes": "short summary"
  }}

When complete, print exactly one line:
AGENTIC_RULE_FULL_DATA_DONE report={report_path}
"""


def _resolve_model(model: str) -> str:
    return _MODEL_ALIASES.get(model, model)


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


def _load_txt_doc(doc_name: str, doc_path: str | Path) -> dict[str, Any]:
    path = Path(doc_path)
    raw = _read_text(path)
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
        "doc_path": str(path.resolve()),
        "text": raw,
        "pages": pages,
        "paragraphs": paragraphs,
        "lines": lines,
    }


def _load_rule_fn(rule_file: Path):
    spec = importlib.util.spec_from_file_location(f"_rule_mod_{rule_file.stem}", str(rule_file))
    if spec is None or spec.loader is None:
        raise ValueError(f"Failed to load rule module: {rule_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, value in vars(mod).items():
        if name.startswith("rule_") and callable(value):
            return value
    raise ValueError(f"No function starting with 'rule_' found in {rule_file}")


def _coerce_span(entry: Any) -> dict[str, Any] | None:
    if isinstance(entry, str):
        text = entry.strip()
        return {"text": text} if text else None
    if not isinstance(entry, dict):
        return None
    text = entry.get("text")
    if text is None:
        return None
    text = str(text)
    if not text.strip():
        return None
    coerced = dict(entry)
    coerced["text"] = text
    return coerced


def _span_key(span: dict[str, Any]) -> tuple[Any, ...]:
    return (
        span.get("page_no"),
        span.get("paragraph_no"),
        span.get("line_no"),
        span.get("text"),
    )


def _sort_key(span: dict[str, Any]) -> tuple[Any, ...]:
    return (
        span.get("page_no", 10**9),
        span.get("paragraph_no", 10**9),
        span.get("line_no", 10**9),
        span.get("text", ""),
    )


def _list_rule_files(rules_dir: Path, rule_names: list[str] | None = None) -> list[Path]:
    if rule_names:
        files = [rules_dir / f"{name}.py" for name in rule_names]
        return [path for path in files if path.exists()]
    return sorted(path for path in rules_dir.glob("rule_*.py") if path.is_file())


def _apply_rule_files(document: dict[str, Any], rule_files: list[Path]) -> dict[str, Any]:
    spans: list[dict[str, Any]] = []
    rules_applied: list[str] = []
    rules_no_hits: list[str] = []
    rule_errors: list[str] = []

    for rule_file in rule_files:
        rule_name = rule_file.stem
        try:
            rule_fn = _load_rule_fn(rule_file)
            raw_result = rule_fn(document)
        except Exception as exc:
            rule_errors.append(f"{rule_name}: {exc}")
            rules_no_hits.append(rule_name)
            continue

        if not isinstance(raw_result, list):
            rules_no_hits.append(rule_name)
            continue

        coerced = [_coerce_span(item) for item in raw_result]
        cleaned = [item for item in coerced if item]
        if cleaned:
            spans.extend(cleaned)
            rules_applied.append(rule_name)
        else:
            rules_no_hits.append(rule_name)

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for span in spans:
        key = _span_key(span)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(span)

    deduped.sort(key=_sort_key)
    retrieved_text = "\n\n".join(span["text"] for span in deduped) if deduped else ""

    return {
        "retrieved_spans": deduped,
        "retrieved_text": retrieved_text,
        "retrieved_token_count": _count_tokens(retrieved_text),
        "rules_applied": rules_applied,
        "rules_no_hits": rules_no_hits,
        "rule_errors": rule_errors,
    }


def _qa_call(question: str, retrieved_text: str, *, model_name: str = "gpt54") -> tuple[str | None, int, int, float]:
    model_mod = importlib.import_module(f"models.{model_name}")
    t0 = time.time()
    response = model_mod.client.chat.completions.create(
        model=model_mod.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": _QA_SYSTEM},
            {"role": "user", "content": f"Passage:\n{retrieved_text}\n\nQuestion: {question}"},
        ],
        max_completion_tokens=500,
        temperature=0.0,
    )
    latency = round(time.time() - t0, 3)
    answer = (response.choices[0].message.content or "").strip() or None
    usage = response.usage
    return (
        answer,
        int(usage.prompt_tokens if usage else 0),
        int(usage.completion_tokens if usage else 0),
        latency,
    )


def _judge_call(
    dataset_name: str,
    question: str,
    ground_truth: Any,
    predicted: Any,
) -> tuple[bool, int, int]:
    if ground_truth is None or predicted is None:
        return False, 0, 0

    model_mod = importlib.import_module("models.gpt54")
    gt_str = json.dumps(ground_truth) if not isinstance(ground_truth, str) else ground_truth
    pred_str = str(predicted)
    response = model_mod.client.chat.completions.create(
        model=model_mod.AZURE_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": _JUDGE_SYSTEM_BY_DATASET.get(dataset_name, _JUDGE_SYSTEM_GENERIC),
            },
            {
                "role": "user",
                "content": f"Question: {question}\nGround Truth: {gt_str}\nPredicted: {pred_str}",
            },
        ],
        max_completion_tokens=10,
        temperature=0.0,
    )
    verdict = (response.choices[0].message.content or "").strip().lower()
    usage = response.usage
    return (
        verdict == "correct",
        int(usage.prompt_tokens if usage else 0),
        int(usage.completion_tokens if usage else 0),
    )


def _clean_stderr(stderr: str) -> str:
    lines = [
        line
        for line in stderr.splitlines()
        if line.strip() != "Reading additional input from stdin..."
    ]
    return "\n".join(lines).strip()[:4000]


def _parse_done_output(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("AGENTIC_RULE_FULL_DATA_DONE"):
            rest = line[len("AGENTIC_RULE_FULL_DATA_DONE") :].strip()
            if rest.startswith("report="):
                return rest[len("report=") :].strip()
    return None


def _parse_codex_events(jsonl_text: str) -> dict[str, Any]:
    usage_totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_output_tokens": 0,
    }
    thread_id: str | None = None
    error_message: str | None = None
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
            for key in usage_totals:
                usage_totals[key] += int(usage.get(key) or 0)
        elif event.get("type") in {"error", "turn.failed"}:
            error_message = event.get("message") or str(event.get("error") or "")

    return {
        "usage_totals": usage_totals,
        "thread_id": thread_id,
        "error_message": error_message,
        "event_count": event_count,
    }


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _relpath(path: Path) -> str:
    try:
        return str(path.relative_to(_ROOT))
    except ValueError:
        return str(path)


def _ledger_load(ledger_path: Path) -> dict[str, Any]:
    return _load_json(ledger_path, {"verify_accuracy_calls": []})


def _ledger_append_verify(ledger_path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    ledger = _ledger_load(ledger_path)
    calls = ledger.setdefault("verify_accuracy_calls", [])
    calls.append(entry)
    _write_json(ledger_path, ledger)
    return ledger


def _manifest_docs(manifest: dict[str, Any]) -> dict[str, Path]:
    return {
        entry["doc_name"]: Path(entry["doc_path"])
        for entry in manifest.get("documents", [])
    }


def _manifest_ground_truth(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        entry["doc_name"]: entry.get("ground_truth")
        for entry in manifest.get("documents", [])
    }


def _verify_accuracy_on_docs(
    *,
    manifest: dict[str, Any],
    doc_names: list[str],
    rule_names: list[str] | None = None,
) -> dict[str, Any]:
    docs_map = _manifest_docs(manifest)
    gt_map = _manifest_ground_truth(manifest)
    rules_dir = Path(manifest["rules_dir"])
    question = manifest["question"]
    dataset_name = manifest["dataset"]
    rule_files = _list_rule_files(rules_dir, rule_names)

    results: list[dict[str, Any]] = []
    total_qa_in = 0
    total_qa_out = 0
    total_judge_in = 0
    total_judge_out = 0
    total_latency = 0.0
    cost_ratios: list[float] = []

    for doc_name in doc_names:
        doc_path = docs_map[doc_name]
        document = _load_txt_doc(doc_name, doc_path)
        applied = _apply_rule_files(document, rule_files)
        answer, qa_in, qa_out, latency = _qa_call(question, applied["retrieved_text"], model_name="gpt54")
        correct, judge_in, judge_out = _judge_call(dataset_name, question, gt_map.get(doc_name), answer)
        doc_tokens = _count_tokens(document["text"])
        cost_ratio = applied["retrieved_token_count"] / doc_tokens if doc_tokens > 0 else 0.0

        total_qa_in += qa_in
        total_qa_out += qa_out
        total_judge_in += judge_in
        total_judge_out += judge_out
        total_latency += latency
        cost_ratios.append(cost_ratio)

        results.append(
            {
                "doc_name": doc_name,
                "answer": answer,
                "correct": correct,
                "retrieved_token_count": applied["retrieved_token_count"],
                "cost_ratio": round(cost_ratio, 6),
                "rules_applied": applied["rules_applied"],
            }
        )

    n = len(results)
    n_correct = sum(1 for item in results if item["correct"])
    accuracy = (n_correct / n) if n else 0.0
    return {
        "doc_names": doc_names,
        "rule_names": [path.stem for path in rule_files],
        "n": n,
        "n_correct": n_correct,
        "accuracy": round(accuracy, 6),
        "avg_cost_ratio": round(sum(cost_ratios) / n, 6) if n else 0.0,
        "qa_input_tokens": total_qa_in,
        "qa_output_tokens": total_qa_out,
        "judge_input_tokens": total_judge_in,
        "judge_output_tokens": total_judge_out,
        "input_tokens_total": total_qa_in + total_judge_in,
        "output_tokens_total": total_qa_out + total_judge_out,
        "latency_seconds": round(total_latency, 3),
        "results": results,
    }


def _summarize_rule_hits(
    *,
    manifest: dict[str, Any],
    rule_name: str,
) -> dict[str, Any]:
    docs_map = _manifest_docs(manifest)
    rules_dir = Path(manifest["rules_dir"])
    rule_files = _list_rule_files(rules_dir, [rule_name])
    if not rule_files:
        raise FileNotFoundError(f"Rule not found: {rule_name}")

    hits: list[dict[str, Any]] = []
    cost_ratios: list[float] = []

    for doc_name, doc_path in docs_map.items():
        document = _load_txt_doc(doc_name, doc_path)
        applied = _apply_rule_files(document, rule_files)
        doc_tokens = _count_tokens(document["text"])
        ratio = applied["retrieved_token_count"] / doc_tokens if doc_tokens > 0 else 0.0
        cost_ratios.append(ratio)
        hits.append(
            {
                "doc_name": doc_name,
                "hit": bool(applied["retrieved_text"]),
                "retrieved_token_count": applied["retrieved_token_count"],
                "cost_ratio": round(ratio, 6),
            }
        )

    return {
        "rule_name": rule_name,
        "hit_count": sum(1 for item in hits if item["hit"]),
        "doc_count": len(hits),
        "avg_cost_ratio": round(sum(cost_ratios) / len(cost_ratios), 6) if hits else 0.0,
        "hits": hits,
    }


def run_rule_gen(
    docs: dict[str, str | Path],
    questions: list[str],
    *,
    labels_by_doc: dict[str, dict[str, Any]] | None = None,
    model: str = "gpt54",
    timeout: int = 3600,
    dataset_name: str = "court",
    split_name: str = "all_docs",
    rules_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    run_stem: str = "q01",
    **_: Any,
) -> dict[str, Any]:
    if len(questions) != 1:
        raise ValueError("Rule generation runs one question per agent session; pass exactly one question")

    question = questions[0]
    question_slug = _make_slug(question)
    labels_by_doc = labels_by_doc or {}

    rules_base = Path(
        rules_dir
        or (
            _ROOT
            / "rules"
            / dataset_name
            / f"agentic_rule_full_data_{model}"
            / split_name
            / f"{run_stem}_{question_slug}"
        )
    )
    results_base = Path(
        results_dir
        or (
            _ROOT
            / "results"
            / dataset_name
            / "rule_gen"
            / f"agentic_rule_full_data_{model}"
            / split_name
        )
    )
    rules_base.mkdir(parents=True, exist_ok=True)
    results_base.mkdir(parents=True, exist_ok=True)

    stem = f"{run_stem}_{question_slug}"
    manifest_path = results_base / f"{stem}.manifest.json"
    ledger_path = results_base / f"{stem}.verify_accuracy_ledger.json"
    report_path = results_base / f"{stem}_rule_gen.json"
    log_path = results_base / f"{stem}.codex.jsonl"
    last_message_path = results_base / f"{stem}.codex.last.txt"

    manifest = {
        "dataset": dataset_name,
        "split": split_name,
        "question": question,
        "question_slug": question_slug,
        "model": _resolve_model(model),
        "rules_dir": str(rules_base.resolve()),
        "ledger_path": str(ledger_path.resolve()),
        "report_path": str(report_path.resolve()),
        "verify_accuracy_budget": _VERIFY_BUDGET,
        "documents": [
            {
                "doc_name": doc_name,
                "doc_path": str(Path(doc_path).resolve()),
                "ground_truth": (labels_by_doc.get(doc_name) or {}).get(question),
            }
            for doc_name, doc_path in sorted(docs.items())
        ],
    }
    _write_json(manifest_path, manifest)
    _write_json(ledger_path, {"verify_accuracy_calls": []})

    codex_bin = shutil.which("codex")
    resolved_model = _resolve_model(model)
    if not codex_bin:
        return {
            "status": "error",
            "model": resolved_model,
            "error_message": "codex CLI not found on PATH",
            "question": question,
            "question_slug": question_slug,
        }

    prompt = _AGENT_PROMPT_TEMPLATE.format(
        question=question,
        question_slug=question_slug,
        manifest_path=str(manifest_path.resolve()),
        rules_dir=str(rules_base.resolve()),
        report_path=str(report_path.resolve()),
        ledger_path=str(ledger_path.resolve()),
        verify_budget=_VERIFY_BUDGET,
    )

    cmd = [
        codex_bin,
        "--ask-for-approval",
        "never",
        "exec",
        "--json",
        "--color",
        "never",
        "--model",
        resolved_model,
        "--cd",
        str(_ROOT),
        "--sandbox",
        "danger-full-access",
        "--output-last-message",
        str(last_message_path),
        prompt,
    ]

    t0 = time.time()
    try:
        res = subprocess.run(
            cmd,
            input="",
            capture_output=True,
            text=True,
            cwd=str(_ROOT),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "model": resolved_model,
            "error_message": f"timed out after {timeout}s",
            "question": question,
            "question_slug": question_slug,
            "rules_dir": _relpath(rules_base),
            "agent_log_path": _relpath(log_path),
            "agent_last_message_path": _relpath(last_message_path),
        }

    latency_rule_gen = round(time.time() - t0, 2)
    stdout = res.stdout or ""
    stderr = _clean_stderr(res.stderr or "")
    jsonl_text = stdout.strip()
    log_path.write_text(jsonl_text + ("\n" if jsonl_text else ""), encoding="utf-8")
    parsed = _parse_codex_events(jsonl_text)

    report_path_from_done = _parse_done_output(last_message_path.read_text(encoding="utf-8", errors="ignore") if last_message_path.exists() else "")
    if report_path_from_done and Path(report_path_from_done).exists():
        report_path = Path(report_path_from_done)

    ledger = _ledger_load(ledger_path)
    verify_calls = ledger.get("verify_accuracy_calls", [])
    verify_input_total = sum(int(entry.get("input_tokens_total", 0) or 0) for entry in verify_calls)
    verify_output_total = sum(int(entry.get("output_tokens_total", 0) or 0) for entry in verify_calls)

    rule_files = _list_rule_files(rules_base)
    report = _load_json(report_path, {})

    rule_gen_input = int(parsed["usage_totals"]["input_tokens"]) + verify_input_total
    rule_gen_output = int(parsed["usage_totals"]["output_tokens"]) + verify_output_total
    rule_gen_cached = int(parsed["usage_totals"]["cached_input_tokens"])
    rule_gen_reasoning = int(parsed["usage_totals"]["reasoning_output_tokens"])

    verify_doc_sets = [tuple(entry.get("doc_names", [])) for entry in verify_calls]
    sample_cost_candidates = [entry.get("avg_cost_ratio") for entry in verify_calls if entry.get("avg_cost_ratio") is not None]
    sample_match_candidates = [entry.get("accuracy") for entry in verify_calls if entry.get("accuracy") is not None]

    metadata = {
        "strategy": "agentic_rule_full_data",
        "phase": "rule_generation",
        "status": "ok" if res.returncode == 0 and rule_files else (f"exit_{res.returncode}" if res.returncode else "error"),
        "dataset": dataset_name,
        "question": question,
        "question_slug": question_slug,
        "model": model,
        "split": split_name,
        "doc_count": len(docs),
        "question_count": 1,
        "rules_dir": _relpath(rules_base),
        "results_report_path": _relpath(report_path),
        "verify_accuracy_ledger_path": _relpath(ledger_path),
        "rule_files": [_relpath(path) for path in rule_files],
        "rules_final": report.get("rules_final", [path.stem for path in rule_files]),
        "num_rules_final": len(report.get("rules_final", [path.stem for path in rule_files])),
        "rule_gen": {
            "latency_seconds": latency_rule_gen,
            "input_tokens": rule_gen_input,
            "output_tokens": rule_gen_output,
            "cached_input_tokens": rule_gen_cached,
            "reasoning_tokens": rule_gen_reasoning,
            "working_sample": report.get("working_sample", sorted({doc for call in verify_calls for doc in call.get("doc_names", [])})),
            "num_working_sample": len(report.get("working_sample", sorted({doc for call in verify_calls for doc in call.get("doc_names", [])}))),
            "num_working_sample_iterations": report.get(
                "num_working_sample_iterations",
                len({doc_set for doc_set in verify_doc_sets if doc_set}),
            ),
            "verify_accuracy_calls": {
                "total": len(verify_calls),
                "input_tokens": verify_input_total,
                "output_tokens": verify_output_total,
            },
            "num_rules_written": len(rule_files),
            "num_rules_rejected": int(report.get("num_rules_rejected", 0) or 0),
            "rules_final": report.get("rules_final", [path.stem for path in rule_files]),
            "num_rules_final": len(report.get("rules_final", [path.stem for path in rule_files])),
            "avg_cost_ratio_sample": (
                float(report.get("avg_cost_ratio_sample"))
                if report.get("avg_cost_ratio_sample") is not None
                else (round(sample_cost_candidates[-1], 6) if sample_cost_candidates else 0.0)
            ),
            "match_rate_sample": (
                float(report.get("match_rate_sample"))
                if report.get("match_rate_sample") is not None
                else (round(sample_match_candidates[-1], 6) if sample_match_candidates else 0.0)
            ),
            "notes": report.get("notes"),
        },
        "agent_log_path": _relpath(log_path),
        "agent_last_message_path": _relpath(last_message_path),
        "stderr": stderr or None,
        "codex_thread_id": parsed["thread_id"],
        "codex_event_count": parsed["event_count"],
        "codex_error_message": parsed["error_message"],
    }
    _write_json(report_path, metadata)
    return metadata


def run_dataset(
    docs: dict[str, str | Path],
    questions: list[str],
    **kwargs: Any,
) -> dict[str, Any]:
    """Backward-compatible alias while this module still lives under src/baseline."""
    return run_rule_gen(docs=docs, questions=questions, **kwargs)


def _cmd_list_docs(args: argparse.Namespace) -> int:
    manifest = _load_json(Path(args.manifest), {})
    docs = manifest.get("documents", [])
    payload = [{"doc_name": entry["doc_name"], "doc_path": entry["doc_path"]} for entry in docs]
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _cmd_read_doc_txt(args: argparse.Namespace) -> int:
    manifest = _load_json(Path(args.manifest), {})
    docs = {entry["doc_name"]: entry for entry in manifest.get("documents", [])}
    entry = docs.get(args.doc_name)
    if not entry:
        raise SystemExit(f"Unknown doc_name: {args.doc_name}")
    text = _read_text(Path(entry["doc_path"]))
    max_chars = args.max_chars or 4000
    print(text[:max_chars])
    return 0


def _cmd_compute_cost(args: argparse.Namespace) -> int:
    manifest = _load_json(Path(args.manifest), {})
    docs_map = _manifest_docs(manifest)
    docs = sorted(args.doc_names) if args.doc_names else sorted(docs_map)
    rules_dir = Path(manifest["rules_dir"])
    rule_files = _list_rule_files(rules_dir, args.rule_names)
    results = []
    ratios = []
    for doc_name in docs:
        document = _load_txt_doc(doc_name, docs_map[doc_name])
        applied = _apply_rule_files(document, rule_files)
        doc_tokens = _count_tokens(document["text"])
        ratio = applied["retrieved_token_count"] / doc_tokens if doc_tokens > 0 else 0.0
        ratios.append(ratio)
        results.append(
            {
                "doc_name": doc_name,
                "retrieved_token_count": applied["retrieved_token_count"],
                "cost_ratio": round(ratio, 6),
                "rules_applied": applied["rules_applied"],
            }
        )
    payload = {
        "doc_count": len(results),
        "rule_names": [path.stem for path in rule_files],
        "avg_cost_ratio": round(sum(ratios) / len(ratios), 6) if ratios else 0.0,
        "results": results,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _cmd_inspect_rule(args: argparse.Namespace) -> int:
    manifest = _load_json(Path(args.manifest), {})
    rules_dir = Path(manifest["rules_dir"])
    rule_file = rules_dir / f"{args.rule_name}.py"
    if not rule_file.exists():
        raise SystemExit(f"Rule not found: {args.rule_name}")
    payload = {
        "rule_name": args.rule_name,
        "file": str(rule_file),
        "code": rule_file.read_text(encoding="utf-8"),
        "stats": _summarize_rule_hits(manifest=manifest, rule_name=args.rule_name),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _cmd_verify_accuracy(args: argparse.Namespace) -> int:
    manifest = _load_json(Path(args.manifest), {})
    ledger_path = Path(manifest["ledger_path"])
    ledger = _ledger_load(ledger_path)
    calls = ledger.get("verify_accuracy_calls", [])
    if len(calls) >= int(manifest.get("verify_accuracy_budget", _VERIFY_BUDGET)):
        raise SystemExit("verify_accuracy budget exhausted")

    docs_map = _manifest_docs(manifest)
    doc_names = sorted(args.doc_names) if args.doc_names else sorted(docs_map)
    result = _verify_accuracy_on_docs(
        manifest=manifest,
        doc_names=doc_names,
        rule_names=args.rule_names,
    )
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    _ledger_append_verify(ledger_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strategy 4 helper CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list-docs")
    p.add_argument("--manifest", required=True)
    p.set_defaults(func=_cmd_list_docs)

    p = sub.add_parser("read-doc-txt")
    p.add_argument("--manifest", required=True)
    p.add_argument("--doc-name", required=True)
    p.add_argument("--max-chars", type=int, default=4000)
    p.set_defaults(func=_cmd_read_doc_txt)

    p = sub.add_parser("compute-cost")
    p.add_argument("--manifest", required=True)
    p.add_argument("--doc-names", nargs="*")
    p.add_argument("--rule-names", nargs="*")
    p.set_defaults(func=_cmd_compute_cost)

    p = sub.add_parser("inspect-rule")
    p.add_argument("--manifest", required=True)
    p.add_argument("--rule-name", required=True)
    p.set_defaults(func=_cmd_inspect_rule)

    p = sub.add_parser("verify-accuracy")
    p.add_argument("--manifest", required=True)
    p.add_argument("--doc-names", nargs="*")
    p.add_argument("--rule-names", nargs="*")
    p.set_defaults(func=_cmd_verify_accuracy)

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
