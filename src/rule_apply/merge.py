"""Apply a set of rules to a document, union the retrieved spans, and call the LLM."""

from __future__ import annotations

import importlib
import importlib.util
import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


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


def rule_apply_merge(
    document: dict,
    rule_names: list[str],
    question_slug: str,
    question: str,
    model_name: str = "gpt54",
    rules_dir: str = "rules/financebench/lsf/single_cluster/llm/gpt54/one_shot",
    output_dir: str = "results/financebench/lsf/single_cluster/llm/gpt54/one_shot/rule_run_merge",
    system_prompt: str | None = None,
    retrieve_only: bool = False,
) -> dict:
    """Apply a set of rules, union the retrieved spans, and call the LLM to answer.

    If `retrieve_only=True`, return after the retrieval step with the span/token
    counts and `predicted_answer=None` — no LLM call. Used by cost-profiling,
    which only needs `retrieved_token_count` (a function of retrieval, not the
    answer model), so it costs nothing."""

    texts: list[dict] = document.get("texts", [])
    text_positions: dict[int, int] = {id(s): i for i, s in enumerate(texts)}

    # Step 1 — Apply each rule and union spans
    all_retrieved: list[dict] = []
    rules_with_hits: list[str] = []
    rules_with_no_hits: list[str] = []
    missing_paths: list[str] = []

    for rule_name in rule_names:
        rule_file = Path(rules_dir) / question_slug / f"{rule_name}.py"
        if not rule_file.exists():
            warnings.warn(f"Rule file not found, skipping: {rule_file}")
            missing_paths.append(str(rule_file))
            rules_with_no_hits.append(rule_name)
            continue
        try:
            rule_fn = _load_rule_fn(rule_file)
            spans = rule_fn(document)
        except Exception as exc:
            warnings.warn(f"Rule {rule_name} raised an error, skipping: {exc}")
            rules_with_no_hits.append(rule_name)
            continue

        if spans:
            rules_with_hits.append(rule_name)
            all_retrieved.extend(spans)
        else:
            rules_with_no_hits.append(rule_name)

    if missing_paths and len(missing_paths) == len(rule_names):
        raise FileNotFoundError(
            "All rule files are missing:\n" + "\n".join(missing_paths)
        )

    num_spans_before_dedup = len(all_retrieved)

    # Deduplicate by index in texts array
    seen_indices: set[int] = set()
    union_spans: list[dict] = []
    for span in all_retrieved:
        idx = text_positions.get(id(span))
        if idx is None:
            try:
                idx = texts.index(span)
            except ValueError:
                idx = None
        if idx is None or idx not in seen_indices:
            if idx is not None:
                seen_indices.add(idx)
            union_spans.append(span)

    num_spans_after_dedup = len(union_spans)

    # Step 2 — Sort in reading order
    def _sort_key(span: dict) -> tuple:
        page = span.get("page_no", 0)
        structure = span.get("structure") or {}
        level_index = structure.get("level_index")
        if level_index is None:
            level_index = text_positions.get(id(span), 0)
        return (page, level_index)

    sorted_spans = sorted(union_spans, key=_sort_key)
    retrieved_text = "\n\n".join(s["text"] for s in sorted_spans) if sorted_spans else ""

    # Step 3 — Count tokens
    retrieved_token_count = _count_tokens(retrieved_text)

    # Cost-profiling shortcut: retrieved_token_count is fully determined by
    # retrieval, so return here without any LLM call (free).
    if retrieve_only:
        return {
            "rule_names": rule_names,
            "question_slug": question_slug,
            "question": question,
            "doc_name": document.get("doc_name", ""),
            "strategy": "merge",
            "predicted_answer": None,
            "retrieved_token_count": retrieved_token_count,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_seconds": 0.0,
        }

    # Cap retrieved text to the model context budget. Large docs (e.g. officeqa)
    # with broad rule pools can produce passages over the model's token limit
    # (~272K), which 400s with context_length_exceeded; truncate so apply
    # degrades gracefully instead of erroring. (Applied only on the LLM path,
    # not retrieve_only, so cost profiling still sees the true retrieval size.)
    _CTX_TOKEN_CAP = 250_000
    if retrieved_token_count > _CTX_TOKEN_CAP and retrieved_text:
        keep = max(1, int(len(retrieved_text) * _CTX_TOKEN_CAP / retrieved_token_count))
        retrieved_text = retrieved_text[:keep]
        retrieved_token_count = _count_tokens(retrieved_text)
        print(f"  [merge] retrieved text exceeded context limit; truncated to "
              f"~{retrieved_token_count} tokens for {document.get('doc_name','')}", flush=True)

    # Step 4 — Call LLM
    model_mod = importlib.import_module(f"models.{model_name}")

    if system_prompt is None:
        system_prompt = (
            "You are a financial document QA assistant.\n"
            "You are given a passage extracted from a financial filing and a question.\n"
            "Answer the question using only the provided passage.\n"
            'If the passage does not contain enough information to answer, reply with "NOT FOUND".\n'
            "Return only the answer — a short value or phrase, not a full sentence."
        )
    user_prompt = f"Passage:\n{retrieved_text}\n\nQuestion: {question}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    t0 = time.time()
    response = model_mod.client.chat.completions.create(
        model=model_mod.AZURE_DEPLOYMENT,
        messages=messages,
        max_completion_tokens=500,
        temperature=0.0,
    )
    latency_seconds = time.time() - t0

    predicted_answer: str | None = (response.choices[0].message.content or "").strip() or None
    usage = response.usage
    input_tokens: int = usage.prompt_tokens if usage else 0
    output_tokens: int = usage.completion_tokens if usage else 0

    # Step 5 — Build result and append to output file
    rule_set_slug = "__".join(sorted(rule_names))[:120]

    result: dict[str, Any] = {
        "rule_names": rule_names,
        "rule_set_slug": rule_set_slug,
        "question_slug": question_slug,
        "question": question,
        "doc_name": document.get("doc_name", ""),
        "strategy": "merge",
        "predicted_answer": predicted_answer,
        "latency_seconds": round(latency_seconds, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "retrieved_token_count": retrieved_token_count,
        "rules_with_hits": rules_with_hits,
        "rules_with_no_hits": rules_with_no_hits,
        "num_spans_before_dedup": num_spans_before_dedup,
        "num_spans_after_dedup": num_spans_after_dedup,
        "retrieved_spans": sorted_spans,
        "retrieved_text": retrieved_text,
    }

    out_path = Path(output_dir) / question_slug / f"{rule_set_slug}_merge.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        existing: list[dict] = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        existing = []

    existing.append(result)
    out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

    return result


if __name__ == "__main__":
    import glob as _glob

    doc_path = _ROOT / "data/financebench/processing/3M_2023Q2_10Q_reconstructed.json"
    document = json.loads(doc_path.read_text(encoding="utf-8"))

    queries_path = _ROOT / "data/financebench/queries.txt"
    first_question = queries_path.read_text(encoding="utf-8").splitlines()[0].strip()

    slug = first_question.lower()
    slug = re.sub(r"[^\w\s]", "", slug)
    slug = re.sub(r"\s+", "_", slug)
    question_slug = slug[:60]

    rule_dirs = sorted(_glob.glob(str(_ROOT / "rules/financebench/lsf/single_cluster/llm/gpt54/one_shot" / f"{question_slug}*")))
    if not rule_dirs:
        print(f"No rule folders found matching rules/financebench/lsf/single_cluster/llm/gpt54/one_shot/{question_slug}*/")
        sys.exit(1)
    rule_dir = rule_dirs[-1]
    folder_slug = Path(rule_dir).name
    rule_files = sorted(_glob.glob(str(Path(rule_dir) / "*.py")))
    if not rule_files:
        print(f"No .py files found in {rule_dir}")
        sys.exit(1)

    rule_names = [Path(f).stem for f in rule_files]

    result = rule_apply_merge(
        document=document,
        rule_names=rule_names,
        question_slug=folder_slug,
        question=first_question,
    )

    print(f"predicted_answer:        {result['predicted_answer']}")
    print(f"retrieved_token_count:   {result['retrieved_token_count']}")
    print(f"num_spans_before_dedup:  {result['num_spans_before_dedup']}")
    print(f"num_spans_after_dedup:   {result['num_spans_after_dedup']}")
    print(f"rules_with_hits:         {len(result['rules_with_hits'])}")
    print(f"rules_with_no_hits:      {len(result['rules_with_no_hits'])}")
