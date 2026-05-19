"""Baseline strategy 1: Agentic Claude QA.

Two modes depending on --model:
  opus47  — spawn a claude -p (Claude Code) session; agent uses Read/Bash tools
            to inspect the reconstructed JSON and answer the question.
  gpt54   — single direct chat-completion call with the full document text.

Usage (single pair):
    python src/baseline/agentic_claude_qa.py \
        --doc data/financebench/processing/JPMORGAN_2023_10K_reconstructed.json \
        --question "What is the registrant's telephone number?" \
        --model opus47
"""

from __future__ import annotations

import argparse
import importlib
import json
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

_DIRECT_SYSTEM = """\
You are a financial document QA assistant. Given the full text of a document and a question,
return only the answer — a short string (number, name, date, address, etc.).
If the answer is not present in the document, reply with exactly: NOT_FOUND"""


def _doc_text(doc_path: str | Path) -> str:
    doc = json.loads(Path(doc_path).read_text(encoding="utf-8"))
    return "\n".join(s.get("text", "") for s in doc.get("texts", []))


def run_opus47(doc_path: str | Path, question: str, timeout: int = 300) -> dict:
    prompt = _AGENT_PROMPT_TEMPLATE.format(
        question=question,
        doc_path=str(Path(doc_path).resolve()),
    )
    cmd = [
        "claude", "--model", _MODEL_ALIASES["opus47"],
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
        "model": _MODEL_ALIASES["opus47"],
    }


def run_gpt54(doc_path: str | Path, question: str) -> dict:
    gpt54 = importlib.import_module("models.gpt54")
    text  = _doc_text(doc_path)
    user_msg = f"Document:\n{text}\n\nQuestion: {question}"

    t0 = time.time()
    resp = gpt54.client.chat.completions.create(
        model=gpt54.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system",  "content": _DIRECT_SYSTEM},
            {"role": "user",    "content": user_msg},
        ],
        temperature=0.0,
    )
    latency = round(time.time() - t0, 2)

    answer        = (resp.choices[0].message.content or "").strip()
    input_tokens  = resp.usage.prompt_tokens     if resp.usage else 0
    output_tokens = resp.usage.completion_tokens if resp.usage else 0

    return {
        "status": "ok",
        "answer": answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_seconds": latency,
        "total_cost_usd": None,
        "model": gpt54.AZURE_DEPLOYMENT,
    }


def run_qa(doc_path: str | Path, question: str, model: str = "opus47",
           timeout: int = 300) -> dict:
    if model in _MODEL_ALIASES or model.startswith("claude"):
        return run_opus47(doc_path, question, timeout=timeout)
    if "gpt" in model or model == "gpt54":
        return run_gpt54(doc_path, question)
    raise ValueError(f"Unknown model: {model!r}. Use one of: {list(_MODEL_ALIASES)} or 'gpt54'")


def main() -> None:
    ap = argparse.ArgumentParser(description="Agentic Claude QA — single pair")
    ap.add_argument("--doc",      required=True, help="Path to reconstructed JSON")
    ap.add_argument("--question", required=True, help="Question to answer")
    ap.add_argument("--model",    default="opus47",
                    help="Model alias: opus47 (default), gpt54, sonnet, haiku")
    ap.add_argument("--timeout",  type=int, default=300,
                    help="Timeout in seconds for claude sessions (default 300)")
    args = ap.parse_args()

    result = run_qa(args.doc, args.question, model=args.model, timeout=args.timeout)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
