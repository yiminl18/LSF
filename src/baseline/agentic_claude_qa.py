"""Baseline strategy 1: Agentic Claude QA.

Spawns a claude -p (Claude Code) session; agent uses Read/Bash tools
to inspect the reconstructed JSON and answer the question.

Usage (single pair):
    python src/baseline/agentic_claude_qa.py \
        --doc data/financebench/processing/JPMORGAN_2023_10K_reconstructed.json \
        --question "What is the registrant's telephone number?"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

_MODEL_ALIASES: dict[str, str] = {
    "opus47": "claude-opus-4-7",
    "opus":   "claude-opus-4-5",
    "sonnet": "claude-sonnet-4-5",
    "haiku":  "claude-haiku-4-5-20251001",
}

_AGENT_PROMPT_TEMPLATE = """\
Answer the following question about the document at the path below.

QUESTION : {question}
DOCUMENT : {doc_path}

Steps:
1. Read the file at DOCUMENT (it is a JSON with a "texts" list; each element has
   "text", "page_no", "size", "bold", "label", and "structure" fields).
2. Scan the spans to locate the answer to QUESTION.
3. Output exactly one line:
   AGENTIC_QA_DONE answer=<your answer>

Rules:
- Give a short, direct answer (a number, name, date, address, etc.) — no explanation.
- If the answer is not found in the document, output:
  AGENTIC_QA_DONE answer=NOT_FOUND
- Do not output anything else after the AGENTIC_QA_DONE line.
"""


def _find_claude() -> str:
    # Search PATH plus common npm-global locations
    extra = [
        os.path.expanduser("~/.npm-global/bin"),
        os.path.expanduser("~/.local/bin"),
        "/usr/local/bin",
    ]
    augmented = os.pathsep.join(extra) + os.pathsep + os.environ.get("PATH", "")
    found = shutil.which("claude", path=augmented)
    return found or "claude"


def run_opus47(doc_path: str | Path, question: str, timeout: int = 300,
               model_alias: str = "opus47") -> dict:
    prompt = _AGENT_PROMPT_TEMPLATE.format(
        question=question,
        doc_path=str(Path(doc_path).resolve()),
    )
    model_id = _MODEL_ALIASES.get(model_alias, _MODEL_ALIASES["opus47"])
    cmd = [
        _find_claude(), "--model", model_id,
        "--output-format", "json",
        "--dangerously-skip-permissions",
        "-p", prompt,
    ]
    t0 = time.time()
    try:
        res = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(_ROOT), timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "answer": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_seconds": round(time.time() - t0, 2),
            "total_cost_usd": None,
            "model": _MODEL_ALIASES["opus47"],
        }

    latency = round(time.time() - t0, 2)
    payload: dict = {}
    try:
        payload = json.loads(res.stdout or "{}")
    except json.JSONDecodeError:
        pass

    usage        = payload.get("usage") or {}
    agent_text   = payload.get("result") or res.stdout or ""
    cost_usd     = payload.get("total_cost_usd")
    input_tokens = (
        (usage.get("input_tokens") or 0)
        + (usage.get("cache_read_input_tokens") or 0)
        + (usage.get("cache_creation_input_tokens") or 0)
    )
    output_tokens = usage.get("output_tokens") or 0

    answer = None
    for line in agent_text.splitlines():
        if line.startswith("AGENTIC_QA_DONE"):
            rest = line[len("AGENTIC_QA_DONE"):].strip()
            if rest.startswith("answer="):
                answer = rest[len("answer="):].strip()
            break

    return {
        "status": "ok" if res.returncode == 0 else f"exit_{res.returncode}",
        "answer": answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_seconds": latency,
        "total_cost_usd": cost_usd,
        "model": model_id,
    }


def run_qa(doc_path: str | Path, question: str, model: str = "opus47",
           timeout: int = 300, **_) -> dict:
    return run_opus47(doc_path, question, timeout=timeout, model_alias=model)


def main() -> None:
    ap = argparse.ArgumentParser(description="Agentic Claude QA — single pair")
    ap.add_argument("--doc",      required=True, help="Path to reconstructed JSON")
    ap.add_argument("--question", required=True, help="Question to answer")
    ap.add_argument("--model",    default="opus47",
                    help="Model alias: opus47 (default), sonnet, haiku")
    ap.add_argument("--timeout",  type=int, default=300,
                    help="Timeout in seconds for claude sessions (default 300)")
    args = ap.parse_args()

    result = run_qa(args.doc, args.question, model=args.model, timeout=args.timeout)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
