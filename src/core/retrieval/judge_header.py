"""
Semantic Match Judgment Module

Provides semantic match verification for QA results, determining whether
model responses match expected answers.

Main functions:
- judge_header(): Judge whether header text can correctly answer a question
- equal_llm(): Use LLM for semantic equivalence checking
- normalize_exact(): Text normalization

Dependencies:
- core.llm.ask: QA functionality
- core.llm.model: Unified LLM dispatch
"""

import argparse
from typing import Optional, Tuple, Any
import hashlib
from pathlib import Path
import sqlite3
import threading

import json

from core.llm.ask import ask
from core.llm.errors import ContentFilterError
from core.llm.model import llm_call, LLM_PROVIDERS
from core.llm.tokens import estimate_tokens
from core.config import JUDGE_CACHE_DIR

# Cache configuration
CACHE_DIR = Path(JUDGE_CACHE_DIR)
CACHE_DB = CACHE_DIR / "judge_cache_v2.db"

# Thread-local storage for DB connections
_thread_local = threading.local()
JUDGE_MODES = {"answer_compare", "support_judge"}

# Prompt template constants (hash used for cache key; cache auto-invalidates when content changes)
_SUPPORT_JUDGE_PROMPT = (
    "Does the context explicitly contain the answer to the question? "
    "The answer must be directly stated in the context text itself, not just implied by the title or section name. "
    "A section title alone (like 'Market Information') is NOT sufficient - the actual answer (e.g., 'ATVI' or 'Nasdaq') must appear in the context. "
    "Return only True or False, nothing else."
)

_ANSWER_COMPARE_PROMPT = (
    "I have two answers to the given question. If these two answers are "
    "equivalent in meaning, return True; otherwise, return False. Ignore minor wording differences. "
    "Do not provide any explanation. "
)

# Prompt hash: 8-char MD5 prefix; cache auto-invalidates when prompt changes
PROMPT_HASHES: dict[str, str] = {
    "support_judge": hashlib.md5(_SUPPORT_JUDGE_PROMPT.encode()).hexdigest()[:8],
    "answer_compare": hashlib.md5(_ANSWER_COMPARE_PROMPT.encode()).hexdigest()[:8],
}


def _get_db_connection():
    """Get a thread-local DB connection."""
    if not hasattr(_thread_local, "connection"):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _thread_local.connection = sqlite3.connect(CACHE_DB)
        with _thread_local.connection:
            _thread_local.connection.execute("""
                CREATE TABLE IF NOT EXISTS judgments (
                    hash TEXT PRIMARY KEY,
                    is_match INTEGER,
                    response TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
    return _thread_local.connection


def _get_cache_key(
    text: str,
    question: str,
    ground_truth: str,
    mode: str,
    llm_provider: str = "azure",
    llm_model: str = "",
    path_text: str = "",
) -> str:
    """Generate a unique key for the inputs.

    Includes prompt_hash so cache auto-invalidates when prompt template changes,
    and llm_provider/llm_model to prevent cross-provider or cross-model cache contamination.
    """
    prompt_hash = PROMPT_HASHES.get(mode, "unknown")
    content = f"{prompt_hash}|{llm_provider}|{llm_model}|{mode}|{text}|{question}|{ground_truth}|{path_text}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


_INVALID_RESPONSE_MARKERS = [
    "missing required environment variable",
    "api_key",
    "authentication",
    "rate limit",
    "rate_limit",
    "connection error",
    "timeout",
    "insufficient_quota",
    "exceeded your current quota",
    "error code: 4",
]


def _check_cache(key: str) -> Any:
    """Check DB for key. Empty responses or error messages are treated as invalid cache and deleted."""
    conn = _get_db_connection()
    cursor = conn.execute(
        "SELECT is_match, response FROM judgments WHERE hash = ?", (key,)
    )
    row = cursor.fetchone()
    if row:
        resp = row[1] or ""
        resp_lower = resp.strip().lower()
        # Empty response or response containing error markers is invalid
        if not resp_lower or any(m in resp_lower for m in _INVALID_RESPONSE_MARKERS):
            with conn:
                conn.execute("DELETE FROM judgments WHERE hash = ?", (key,))
            return None
        return (bool(row[0]), row[1])
    return None


def _save_to_cache(key: str, value: Tuple[bool, str]):
    """Save to DB."""
    conn = _get_db_connection()
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO judgments (hash, is_match, response) VALUES (?, ?, ?)",
            (key, int(value[0]), value[1]),
        )


def normalize_exact(s: str) -> str:
    """Text normalization: lowercase, strip whitespace and simple punctuation."""
    if not s:
        return ""
    cleaned = s.strip().lower()
    cleaned = cleaned.replace(".", "").replace(",", "")
    return cleaned


def to_text(value) -> str:
    """Convert non-string ground truth/response to string for comparison."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _require_llm_model(llm_model: Optional[str]) -> str:
    if not isinstance(llm_model, str) or not llm_model.strip():
        raise ValueError("llm_model must be specified explicitly")
    return llm_model.strip()


def _judge_support(
    text: str,
    question: str,
    ground_truth: str,
    llm_provider: str = "azure",
    *,
    llm_model: str,
    path_text: str = "",
) -> Tuple[bool, str]:
    """Directly ask LLM whether context contains the given answer (single call, lower cost)."""
    # Build context with path information
    context_parts = []
    if path_text:
        context_parts.append(f"Document Path: {path_text}")
    context_parts.append(f"Context:\n{text}")
    context_str = "\n\n".join(context_parts)

    instruction = (
        f"{_SUPPORT_JUDGE_PROMPT}\n\n"
        f"{context_str}\n\n"
        f"Question: {question}\n"
        f"Answer: {ground_truth}\n\n"
        "Does the context explicitly contain this answer? (True/False):"
    )
    try:
        resp = llm_call(
            instruction,
            llm_provider=llm_provider,
            model=llm_model,
            max_tokens=10,
        )
        resp_str = resp if isinstance(resp, str) else str(resp)
        resp_lower = resp_str.lower().strip()
        # Strict check: must explicitly be "true", not just contain the word "true"
        # Check if starts with "true" (after stripping), or first word is "true"
        words = resp_lower.split()
        is_true = resp_lower.startswith("true") and (
            len(words) == 0 or words[0] == "true"
        )
        return (is_true, resp_str)
    except ContentFilterError:
        raise
    except Exception as e:
        return (False, str(e))


def equal_llm(
    res1: str,
    res2: str,
    question: str,
    llm_provider: str = "azure",
    *,
    llm_model: str,
) -> Tuple[bool, int]:
    """
    Use LLM to check semantic equivalence of two answers.

    Args:
        res1: First answer
        res2: Second answer
        question: Question text
        llm_provider: LLM provider

    Returns:
        Tuple of (is_equivalent, estimated_input_tokens)
    """
    instruction = (
        _ANSWER_COMPARE_PROMPT
        + "Answer 1: "
        + str(res1)
        + " Answer 2: "
        + str(res2)
        + " Question: "
        + str(question)
    )
    tokens = estimate_tokens(instruction, model=llm_model)
    try:
        resp = llm_call(
            instruction,
            llm_provider=llm_provider,
            model=llm_model,
            max_tokens=8,
        )
        resp_str = resp if isinstance(resp, str) else str(resp)
        if "true" in resp_str.lower():
            return True, tokens
    except Exception:
        pass
    return False, tokens


def judge_header(
    text: str,
    question: str,
    ground_truth: str,
    mode: str = "answer_compare",
    llm_provider: str = "azure",
    llm_model: Optional[str] = None,
    path_text: str = "",
) -> Tuple[bool, str]:
    """
    Judge whether header text can correctly answer a question.

    Calls ask() to get a response, then performs normalized matching:
      1) If normalized text is empty or "none", no match
      2) If normalized texts are exactly equal, match
      3) Otherwise use equal_llm for semantic matching

    Args:
        text: Header text (context)
        question: Question text
        ground_truth: Expected correct answer
        mode: Judge mode ("answer_compare" or "support_judge")
        llm_provider: LLM provider ("azure" or "openrouter")
        llm_model: LLM model name
        path_text: Document path info (optional, provides extra context for support_judge mode)

    Returns:
        Tuple of (is_match, raw_response)
    """
    if mode not in JUDGE_MODES:
        raise ValueError(
            f"Unsupported judge mode={mode}. Expected one of {sorted(JUDGE_MODES)}"
        )
    resolved_model = _require_llm_model(llm_model)

    # 0. Check Cache (SQLite), scoped by provider and model.
    cache_key = _get_cache_key(
        text,
        question,
        ground_truth,
        mode,
        llm_provider,
        resolved_model,
        path_text,
    )
    cached_val = _check_cache(cache_key)
    if cached_val:
        print(
            f"[JUDGE_CACHE_HIT] mode={mode} match={cached_val[0]} resp={cached_val[1]!r}"
        )
        return cached_val

    # support_judge: single LLM call, directly judge whether context supports the answer
    if mode == "support_judge":
        result = _judge_support(
            text,
            question,
            ground_truth,
            llm_provider=llm_provider,
            llm_model=resolved_model,
            path_text=path_text,
        )
        _save_to_cache(cache_key, result)
        return result

    # answer_compare: ask() + normalized matching + equal_llm()
    try:
        predicted_raw = ask(
            text,
            question,
            llm_provider=llm_provider,
            model=resolved_model,
        )
        predicted = to_text(predicted_raw)
        # Ensure predicted is a string (to_text should always return str, but just in case)
        if not isinstance(predicted, str):
            predicted = str(predicted) if predicted is not None else ""
    except ContentFilterError:
        raise
    except Exception as e:
        # Check if this is a content filter error (jailbreak detection)
        error_msg = str(e)
        if (
            "content_filter" in error_msg.lower()
            or "content management policy" in error_msg.lower()
            or "jailbreak" in error_msg.lower()
        ):
            # Content filter error: raise immediately to avoid further API calls
            raise ContentFilterError(
                f"Content filter triggered (jailbreak detected). "
                f"Text preview: {text[:200]}... Question: {question[:100]}... "
                f"Original error: {error_msg[:500]}"
            ) from e
        else:
            # Other errors: re-raise
            raise

    gt_text = to_text(ground_truth) if ground_truth is not None else ""
    predicted_norm = normalize_exact(predicted)
    gt_norm = normalize_exact(gt_text)

    result = (False, predicted)  # Default

    # 1) Normalized text comparison (including the case where both are "none")
    if predicted_norm and gt_norm and predicted_norm == gt_norm:
        result = (True, predicted)

    # 2) If predicted is "None" or empty, decide directly (no need to call equal_llm)
    elif not predicted_norm or predicted_norm == "none":
        # If ground_truth is also "None" or empty, it matches (but that case should
        # already be handled in step 1). Otherwise no match, return False directly.
        result = (False, predicted)

    # 3) Semantic equivalence check (only called when predicted is not "None", saves cost)
    else:
        # predicted is already ensured to be a string, but use or "" as fallback for safety
        is_eq, _ = equal_llm(
            predicted or "",
            ground_truth,
            question,
            llm_provider=llm_provider,
            llm_model=resolved_model,
        )
        result = (is_eq, predicted)

    # 4. Update Cache (SQLite)
    _save_to_cache(cache_key, result)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge a single header text against a question/answer."
    )
    parser.add_argument(
        "--text", required=True, help="Header text (or combined text_span)."
    )
    parser.add_argument("--question", required=True, help="Question string.")
    parser.add_argument("--answer", required=True, help="Ground-truth answer string.")
    parser.add_argument(
        "--mode",
        default="answer_compare",
        choices=sorted(JUDGE_MODES),
        help="judge mode",
    )
    parser.add_argument(
        "--llm-provider",
        default="azure",
        choices=sorted(LLM_PROVIDERS),
        help="LLM provider",
    )
    parser.add_argument("--model", required=True, help="LLM model")
    args = parser.parse_args()

    matched, resp = judge_header(
        args.text,
        args.question,
        args.answer,
        mode=args.mode,
        llm_provider=args.llm_provider,
        llm_model=args.model,
    )
    print(f"matched={matched}")
    print(f"llm_response={resp}")


if __name__ == "__main__":
    main()
