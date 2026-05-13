"""Phase 0: compute per-rule token costs without LLM calls.

Loads each rule, applies it to every document, and counts tokens in the
retrieved text using tiktoken.  No QA or judge LLM calls are made.

Output schema per rule:
    {
        "W": <total retrieved tokens across all docs>,
        "per_doc": {doc_name: retrieved_token_count, ...},
        "per_doc_ratio": {doc_name: retrieved / total_doc_tokens, ...},
        "avg_cost_ratio": <mean of per_doc_ratio values>
    }
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


def _load_rule_fn(rule_file: Path):
    spec = importlib.util.spec_from_file_location("_rule_mod", str(rule_file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        return next(v for k, v in vars(mod).items() if k.startswith("rule_") and callable(v))
    except StopIteration:
        raise ValueError(f"No function starting with 'rule_' found in {rule_file}")


def compute_rule_costs(
    rules_dir: str,
    doc_paths: list[Path],
) -> dict[str, dict]:
    """Return cost profile for every rule_*.py in rules_dir. No LLM calls."""
    rules_path = Path(rules_dir)
    rule_files = sorted(rules_path.glob("rule_*.py"))
    if not rule_files:
        raise FileNotFoundError(f"No rule_*.py files found in {rules_dir}")

    result: dict[str, dict] = {}

    for rule_file in rule_files:
        rule_name = rule_file.stem
        try:
            rule_fn = _load_rule_fn(rule_file)
        except Exception as exc:
            print(f"  WARNING: skipping {rule_name}: {exc}")
            continue

        per_doc: dict[str, int] = {}
        per_doc_ratio: dict[str, float] = {}
        W = 0

        for doc_path in doc_paths:
            doc_name = doc_path.stem.replace("_reconstructed", "")
            try:
                doc = json.loads(doc_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            total_doc_tokens = _count_tokens(
                "\n".join(s.get("text", "") for s in doc.get("texts", []))
            )

            try:
                spans = rule_fn(doc)
            except Exception:
                spans = []

            retrieved_text = "\n\n".join(s["text"] for s in spans) if spans else ""
            retrieved_tokens = _count_tokens(retrieved_text)

            per_doc[doc_name] = retrieved_tokens
            W += retrieved_tokens
            if total_doc_tokens > 0:
                per_doc_ratio[doc_name] = retrieved_tokens / total_doc_tokens

        avg_cost_ratio = (
            sum(per_doc_ratio.values()) / len(per_doc_ratio) if per_doc_ratio else 0.0
        )

        result[rule_name] = {
            "W": W,
            "per_doc": per_doc,
            "per_doc_ratio": per_doc_ratio,
            "avg_cost_ratio": round(avg_cost_ratio, 6),
        }

    return result


def load_or_compute_cost_profile(
    rules_dir: str,
    doc_paths: list[Path],
    cache_path: Path | None = None,
) -> dict[str, dict]:
    """Return cost profile from cache_path if it exists, else compute and optionally cache."""
    if cache_path is not None and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    profile = compute_rule_costs(rules_dir, doc_paths)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return profile
