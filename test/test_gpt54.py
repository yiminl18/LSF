"""Quick smoke-test for models/gpt54.py

Run from the repo root:
    python test/test_gpt54.py

Exits with code 0 on success, 1 on any failure.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make sure src/ is on the path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from models import gpt54 as _gpt


# ─── helpers ──────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _ok(label: str, value: str) -> None:
    preview = value[:120].replace("\n", " ")
    print(f"  [PASS] {label}: {preview!r}")


def _fail(label: str, err: Exception) -> None:
    print(f"  [FAIL] {label}: {err}")


# ─── tests ────────────────────────────────────────────────────────────────────

def test_config() -> bool:
    """Print loaded Azure credentials (key is partially masked)."""
    _section("1. Config check")
    try:
        print(f"  endpoint   : {_gpt.AZURE_ENDPOINT}")
        print(f"  api_version: {_gpt.AZURE_API_VERSION}")
        print(f"  deployment : {_gpt.AZURE_DEPLOYMENT}")
        key = _gpt.api_key
        masked = (key[:8] + "…") if len(key) > 8 else "***"
        print(f"  api_key    : {masked}")
        return True
    except Exception as e:
        _fail("config", e)
        return False


def test_chat_completions() -> bool:
    """Test the low-level chat_completions() call."""
    _section("2. chat_completions() — basic call")
    try:
        t0 = time.perf_counter()
        response = _gpt.chat_completions(
            "Reply with exactly three words: Hello from GPT.",
            system="You are a concise assistant.",
            max_completion_tokens=32,
        )
        elapsed = time.perf_counter() - t0
        assert isinstance(response, str) and len(response) > 0, "empty response"
        _ok(f"response ({elapsed:.2f}s)", response)
        return True
    except Exception as e:
        _fail("chat_completions", e)
        return False


def test_gpt_54_with_context() -> bool:
    """Test the higher-level gpt_54() QA call."""
    _section("3. gpt_54() — QA with context")
    context = (
        "Acme Corp was founded in 1998 in San Francisco. "
        "Its annual revenue for fiscal year 2024 was $4.2 billion. "
        "The CEO is Jane Smith."
    )
    question = "What was Acme Corp's revenue in fiscal year 2024?"
    expected_fragment = "4.2"

    try:
        t0 = time.perf_counter()
        answer = _gpt.gpt_54(question, context, max_completion_tokens=128)
        elapsed = time.perf_counter() - t0
        assert isinstance(answer, str) and len(answer) > 0, "empty answer"
        assert expected_fragment in answer, (
            f"expected '{expected_fragment}' in answer, got: {answer!r}"
        )
        _ok(f"answer ({elapsed:.2f}s)", answer)
        return True
    except Exception as e:
        _fail("gpt_54", e)
        return False


def test_gpt_54_no_context() -> bool:
    """Verify the model politely says it lacks info when context is empty."""
    _section("4. gpt_54() — empty context (should decline)")
    try:
        t0 = time.perf_counter()
        answer = _gpt.gpt_54(
            "What is the stock price of XYZ Corp today?",
            context="",
            max_completion_tokens=64,
        )
        elapsed = time.perf_counter() - t0
        assert isinstance(answer, str) and len(answer) > 0, "empty answer"
        _ok(f"answer ({elapsed:.2f}s)", answer)
        return True
    except Exception as e:
        _fail("gpt_54 (no context)", e)
        return False


# ─── runner ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("\nRunning gpt54 smoke tests …")
    results = [
        test_config(),
        test_chat_completions(),
        test_gpt_54_with_context(),
        test_gpt_54_no_context(),
    ]

    passed = sum(results)
    total  = len(results)
    _section(f"Summary: {passed}/{total} passed")

    if passed < total:
        sys.exit(1)
    print("\n  All tests passed.\n")


if __name__ == "__main__":
    main()
