"""Dataset-scope Agentic Codex QA baseline.

Runs one non-interactive Codex CLI session over a manifest containing every
selected document and every selected question. The agent may use its default
tools however it wants, but must eventually materialize answers for all
(question, document) pairs into a JSON file that the runner can turn into the
standard per-pair baseline artifacts.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

_MODEL_ALIASES: dict[str, str] = {
    "gpt54": "gpt-5.4",
    "gpt54mini": "gpt-5.4-mini",
    "gpt-5.4": "gpt-5.4",
    "gpt-5.4-mini": "gpt-5.4-mini",
}

_AGENT_PROMPT_TEMPLATE = """\
Complete the dataset-wide QA task described by the manifest below.

MANIFEST : {manifest_path}
OUTPUT   : {answers_path}

You are running as a Codex agent in this repository. Use your default tools.
The manifest lists every document path and every question you must answer.
The document paths point to plain-text `.txt` files.

Goal:
- produce one answer for every question on every document
- inspect any documents you need using whatever loops, iterations, searches,
  planning, or tool calls you want
- write the final answers JSON to OUTPUT

Required output JSON schema at OUTPUT:
{{
  "answers": {{
    "<doc_name>": {{
      "<question 1>": "<answer>",
      "<question 2>": "<answer>",
      "...": "..."
    }},
    "...": {{}}
  }}
}}

Requirements:
- include every listed document exactly once in "answers"
- include every listed question exactly once per document
- use short direct answers only
- if an answer is not found, use exactly "NOT_FOUND"
- inspect the documents themselves; do not use any ground-truth label files
- you may overwrite OUTPUT as many times as you want while working
- do not edit repository files other than OUTPUT
- when complete, print exactly one line:
  AGENTIC_QA_ALL_DONE output={answers_path}
"""


def _resolve_model(model: str) -> str:
    return _MODEL_ALIASES.get(model, model)


def _parse_done_output(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("AGENTIC_QA_ALL_DONE"):
            rest = line[len("AGENTIC_QA_ALL_DONE"):].strip()
            if rest.startswith("output="):
                return rest[len("output="):].strip()
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


def _clean_stderr(stderr: str) -> str:
    lines = [
        line
        for line in stderr.splitlines()
        if line.strip() != "Reading additional input from stdin..."
    ]
    return "\n".join(lines).strip()[:2000]


def _normalize_answers(raw: Any) -> dict[str, dict[str, str]]:
    if isinstance(raw, dict) and isinstance(raw.get("answers"), dict):
        raw = raw["answers"]

    if isinstance(raw, dict):
        normalized: dict[str, dict[str, str]] = {}
        for doc_name, answers in raw.items():
            if not isinstance(doc_name, str):
                continue
            if not isinstance(answers, dict):
                continue
            normalized[doc_name] = {
                str(question): str(answer)
                for question, answer in answers.items()
            }
        return normalized

    if isinstance(raw, list):
        normalized = {}
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            doc_name = entry.get("doc_name")
            answers = entry.get("answers")
            if not isinstance(doc_name, str) or not isinstance(answers, dict):
                continue
            normalized[doc_name] = {
                str(question): str(answer)
                for question, answer in answers.items()
            }
        return normalized

    return {}


def run_dataset(
    docs: dict[str, str | Path],
    questions: list[str],
    *,
    model: str = "gpt54",
    timeout: int = 3600,
    log_dir: str | Path | None = None,
    run_stem: str = "all_docs",
    dataset_name: str = "dataset",
    split_name: str = "all_docs",
) -> dict[str, Any]:
    codex_bin = shutil.which("codex")
    resolved_model = _resolve_model(model)
    if not codex_bin:
        return {
            "status": "error",
            "answers_by_doc": {},
            "input_tokens_total": 0,
            "output_tokens_total": 0,
            "cached_input_tokens_total": 0,
            "reasoning_output_tokens_total": 0,
            "latency_seconds_total": 0.0,
            "total_cost_usd": None,
            "model": resolved_model,
            "error_message": "codex CLI not found on PATH",
        }

    log_base = Path(log_dir or (_ROOT / "logs"))
    log_base.mkdir(parents=True, exist_ok=True)
    manifest_path = log_base / f"{run_stem}.manifest.json"
    answers_path = log_base / f"{run_stem}.answers.json"
    log_path = log_base / f"{run_stem}.codex.jsonl"
    last_message_path = log_base / f"{run_stem}.codex.last.txt"

    manifest = {
        "dataset": dataset_name,
        "split": split_name,
        "model": resolved_model,
        "questions": questions,
        "documents": [
            {
                "doc_name": doc_name,
                "doc_path": str(Path(doc_path).resolve()),
            }
            for doc_name, doc_path in docs.items()
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    prompt = _AGENT_PROMPT_TEMPLATE.format(
        manifest_path=str(manifest_path.resolve()),
        answers_path=str(answers_path.resolve()),
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
        latency = round(time.time() - t0, 2)
        return {
            "status": "timeout",
            "answers_by_doc": {},
            "input_tokens_total": 0,
            "output_tokens_total": 0,
            "cached_input_tokens_total": 0,
            "reasoning_output_tokens_total": 0,
            "latency_seconds_total": latency,
            "total_cost_usd": None,
            "model": resolved_model,
            "manifest_path": str(manifest_path.relative_to(_ROOT)),
            "answers_output_path": str(answers_path.relative_to(_ROOT)),
            "codex_log_path": str(log_path.relative_to(_ROOT)),
            "codex_last_message_path": str(last_message_path.relative_to(_ROOT)),
        }

    latency = round(time.time() - t0, 2)
    jsonl_text = res.stdout or ""
    log_path.write_text(jsonl_text, encoding="utf-8")

    last_text = ""
    if last_message_path.exists():
        try:
            last_text = last_message_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            last_text = ""

    parsed = _parse_codex_events(jsonl_text)
    done_output = _parse_done_output(last_text) or _parse_done_output(jsonl_text)

    answers_by_doc: dict[str, dict[str, str]] = {}
    answers_error: str | None = None
    if answers_path.exists():
        try:
            raw_answers = json.loads(answers_path.read_text(encoding="utf-8"))
            answers_by_doc = _normalize_answers(raw_answers)
        except Exception as exc:
            answers_error = f"Failed to parse answers JSON: {exc}"
    else:
        answers_error = f"Answers file not found: {answers_path}"

    usage = parsed["usage_totals"]
    status = "ok" if res.returncode == 0 else f"exit_{res.returncode}"
    if answers_error and status == "ok":
        status = "error"

    error_message = answers_error or parsed["error_message"]

    return {
        "status": status,
        "answers_by_doc": answers_by_doc,
        "input_tokens_total": int(usage.get("input_tokens") or 0),
        "output_tokens_total": int(usage.get("output_tokens") or 0),
        "cached_input_tokens_total": int(usage.get("cached_input_tokens") or 0),
        "reasoning_output_tokens_total": int(usage.get("reasoning_output_tokens") or 0),
        "latency_seconds_total": latency,
        "total_cost_usd": None,
        "model": resolved_model,
        "codex_thread_id": parsed["thread_id"],
        "codex_event_count": parsed["event_count"],
        "codex_error_message": parsed["error_message"],
        "error_message": error_message,
        "stderr": _clean_stderr(res.stderr or ""),
        "manifest_path": str(manifest_path.relative_to(_ROOT)),
        "answers_output_path": str(answers_path.relative_to(_ROOT)),
        "codex_log_path": str(log_path.relative_to(_ROOT)),
        "codex_last_message_path": str(last_message_path.relative_to(_ROOT)),
        "done_output_path": done_output,
    }
