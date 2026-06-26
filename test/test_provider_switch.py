"""Smoke-test the LLM provider switch (azure vs openai).

Routes the shared model modules (gpt54 / gpt54mini / embedding3small) through the
provider chosen by ``LSF_LLM_PROVIDER`` (env) or ``--provider`` (sets the env
before import), then exercises chat-large, chat-mini, and embeddings.

Usage:
    python3 test/test_provider_switch.py                # current provider (env or azure.json)
    python3 test/test_provider_switch.py --provider openai
    python3 test/test_provider_switch.py --provider azure
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def run() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", choices=["openai", "azure"],
                    help="override LSF_LLM_PROVIDER for this run")
    args = ap.parse_args()
    if args.provider:
        os.environ["LSF_LLM_PROVIDER"] = args.provider

    # Import AFTER setting the env var (clients are built at import time).
    from models import gpt54, gpt54mini, embedding3small

    print(f"requested provider : {os.environ.get('LSF_LLM_PROVIDER', '(default azure)')}")
    print(f"gpt54      -> {gpt54.PROVIDER:6s} / {gpt54.AZURE_DEPLOYMENT}")
    print(f"gpt54mini  -> {gpt54mini.PROVIDER:6s} / {gpt54mini.AZURE_DEPLOYMENT}")
    print(f"embedding  -> {embedding3small.PROVIDER:6s} / {embedding3small.AZURE_DEPLOYMENT}")
    print("-" * 60)

    results: dict[str, bool] = {}

    # 1) chat-large (gpt-5.4)
    try:
        ans = gpt54.chat_completions("Reply with exactly one word: pong.",
                                     max_completion_tokens=16)
        ok = "pong" in ans.lower()
        results["gpt54"] = ok
        print(f"[{'OK ' if ok else 'WARN'}] gpt54      -> {ans!r}")
    except Exception as exc:  # noqa: BLE001
        results["gpt54"] = False
        print(f"[FAIL] gpt54      -> {type(exc).__name__}: {str(exc)[:160]}")

    # 2) chat-mini (gpt-5.4-mini)
    try:
        ans = gpt54mini.chat_completions("Reply with exactly one word: pong.",
                                         max_completion_tokens=16)
        ok = "pong" in ans.lower()
        results["gpt54mini"] = ok
        print(f"[{'OK ' if ok else 'WARN'}] gpt54mini  -> {ans!r}")
    except Exception as exc:  # noqa: BLE001
        results["gpt54mini"] = False
        print(f"[FAIL] gpt54mini  -> {type(exc).__name__}: {str(exc)[:160]}")

    # 3) embeddings
    try:
        vecs = embedding3small.embed(["hello world", "the quick brown fox"])
        ok = len(vecs) == 2 and len(vecs[0]) > 0
        results["embedding"] = ok
        print(f"[{'OK ' if ok else 'WARN'}] embedding  -> {len(vecs)} vectors, dim={len(vecs[0])}")
    except Exception as exc:  # noqa: BLE001
        results["embedding"] = False
        print(f"[FAIL] embedding  -> {type(exc).__name__}: {str(exc)[:160]}")

    print("-" * 60)
    n_ok = sum(results.values())
    print(f"summary: {n_ok}/{len(results)} OK  (provider={gpt54.PROVIDER})")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(run())
