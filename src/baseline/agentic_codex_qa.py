"""Baseline strategy 2: Agentic Codex QA.

Spawns a non-interactive Codex CLI session driven by GPT-5.4. The agent uses
Codex's default tool environment to inspect the reconstructed JSON and answer
the question.

Usage (single pair):
    python src/baseline/agentic_codex_qa.py \
        --doc data/financebench/processing/JPMORGAN_2023_10K_reconstructed.json \
        --question "What is the registrant's telephone number?"
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

_MODEL_ALIASES: dict[str, str] = {
    "gpt54": "gpt-5.4",
    "gpt-5.4": "gpt-5.4",
}

_AGENT_PROMPT_TEMPLATE = """\
Answer the following question about the document at the path below.

QUESTION : {question}
DOCUMENT : {doc_path}

You are running as a Codex agent in this repository. Use your default tools to
inspect DOCUMENT. It is a reconstructed JSON with a "texts" list; each element
has fields such as "text", "page_no", "size", "bold", "label", and "structure".

Steps:
1. Read and search the document JSON as needed.
2. Locate the shortest document-supported answer to QUESTION.
3. Output exactly one line:
   AGENTIC_QA_DONE answer=<your answer>

Rules:
- Give a short, direct answer (a number, name, date, address, etc.) with no explanation.
- If the answer is not found in the document, output:
  AGENTIC_QA_DONE answer=NOT_FOUND
- Do not edit files.
- Do not output anything else after the AGENTIC_QA_DONE line.
"""


def _resolve_model(model: str) -> str:
    return _MODEL_ALIASES.get(model, model)


def _parse_answer(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("AGENTIC_QA_DONE"):
            rest = line[len("AGENTIC_QA_DONE"):].strip()
            if rest.startswith("answer="):
                return rest[len("answer="):].strip()
    return None


def _parse_codex_events(jsonl_text: str) -> dict[str, Any]:
    usage: dict[str, int] = {}
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
        elif event.get("type") in {"error", "turn.failed"}:
            error_message = event.get("message") or str(event.get("error") or "")

    return {
        "usage": usage,
        "thread_id": thread_id,
        "error_message": error_message,
        "event_count": event_count,
    }


def _clean_stderr(stderr: str) -> str:
    lines = [
        line for line in stderr.splitlines()
        if line.strip() != "Reading additional input from stdin..."
    ]
    return "\n".join(lines).strip()[:2000]


def run_codex_gpt54(
    doc_path: str | Path,
    question: str,
    *,
    model: str = "gpt54",
    timeout: int = 300,
    log_dir: str | Path | None = None,
    log_stem: str | None = None,
) -> dict:
    codex_bin = shutil.which("codex")
    resolved_model = _resolve_model(model)
    if not codex_bin:
        return {
            "status": "error",
            "answer": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_seconds": 0.0,
            "total_cost_usd": None,
            "model": resolved_model,
            "error_message": "codex CLI not found on PATH",
        }

    prompt = _AGENT_PROMPT_TEMPLATE.format(
        question=question,
        doc_path=str(Path(doc_path).resolve()),
    )

    log_path: Path | None = None
    last_message_path: Path | None = None
    if log_dir is not None and log_stem:
        log_base = Path(log_dir)
        log_base.mkdir(parents=True, exist_ok=True)
        log_path = log_base / f"{log_stem}.codex.jsonl"
        last_message_path = log_base / f"{log_stem}.codex.last.txt"

    with tempfile.NamedTemporaryFile("w+", delete=False) as last_tmp:
        tmp_last_path = Path(last_tmp.name)

    cmd = [
        codex_bin,
        "--ask-for-approval", "never",
        "exec",
        "--json",
        "--color", "never",
        "--model", resolved_model,
        "--cd", str(_ROOT),
        "--sandbox", "danger-full-access",
        "--output-last-message", str(tmp_last_path),
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
        try:
            tmp_last_path.unlink(missing_ok=True)
        except Exception:
            pass
        return {
            "status": "timeout",
            "answer": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_seconds": latency,
            "total_cost_usd": None,
            "model": resolved_model,
        }

    latency = round(time.time() - t0, 2)
    jsonl_text = res.stdout or ""
    last_text = ""
    try:
        last_text = tmp_last_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        pass
    finally:
        try:
            tmp_last_path.unlink(missing_ok=True)
        except Exception:
            pass

    if log_path is not None:
        log_path.write_text(jsonl_text, encoding="utf-8")
    if last_message_path is not None:
        last_message_path.write_text(last_text, encoding="utf-8")

    parsed = _parse_codex_events(jsonl_text)
    usage = parsed["usage"]
    answer = _parse_answer(last_text) or _parse_answer(jsonl_text)

    direct_input_tokens = int(usage.get("input_tokens") or 0)
    cached_input_tokens = int(usage.get("cached_input_tokens") or 0)
    input_tokens = direct_input_tokens + cached_input_tokens
    output_tokens = int(usage.get("output_tokens") or 0)

    return {
        "status": "ok" if res.returncode == 0 else f"exit_{res.returncode}",
        "answer": answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_seconds": latency,
        "total_cost_usd": None,
        "model": resolved_model,
        "cached_input_tokens": cached_input_tokens,
        "reasoning_output_tokens": int(usage.get("reasoning_output_tokens") or 0),
        "codex_thread_id": parsed["thread_id"],
        "codex_event_count": parsed["event_count"],
        "codex_error_message": parsed["error_message"],
        "codex_log_path": str(log_path.relative_to(_ROOT)) if log_path else None,
        "codex_last_message_path": (
            str(last_message_path.relative_to(_ROOT)) if last_message_path else None
        ),
        "stderr": _clean_stderr(res.stderr or ""),
    }


def run_qa(
    doc_path: str | Path,
    question: str,
    model: str = "gpt54",
    timeout: int = 300,
    **kwargs: Any,
) -> dict:
    return run_codex_gpt54(
        doc_path,
        question,
        model=model,
        timeout=timeout,
        log_dir=kwargs.get("log_dir"),
        log_stem=kwargs.get("log_stem"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Agentic Codex QA - single pair")
    ap.add_argument("--doc", required=True, help="Path to reconstructed JSON")
    ap.add_argument("--question", required=True, help="Question to answer")
    ap.add_argument("--model", default="gpt54", help="Model alias: gpt54 (default)")
    ap.add_argument("--timeout", type=int, default=300,
                    help="Timeout in seconds for Codex sessions (default 300)")
    args = ap.parse_args()

    result = run_qa(args.doc, args.question, model=args.model, timeout=args.timeout)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
