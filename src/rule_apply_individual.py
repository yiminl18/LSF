"""Apply a single named rule to a document and call the LLM to answer a question."""

from __future__ import annotations

import importlib
import importlib.util
import json
import re
import sys
import time
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
    if not rule_file.exists():
        raise FileNotFoundError(
            f"Rule file not found: {rule_file}. "
            f"Generate rules first with rule_gen_llm_coarse."
        )
    spec = importlib.util.spec_from_file_location("_rule_mod", str(rule_file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        return next(v for k, v in vars(mod).items() if k.startswith("rule_") and callable(v))
    except StopIteration:
        raise ValueError(f"No function starting with 'rule_' found in {rule_file}")


def rule_apply_individual(
    document: dict,
    rule_name: str,
    question_slug: str,
    question: str,
    model_name: str = "gpt54",
    rules_dir: str = "rules/financebench_single_cluster/llm/gpt54/one_shot",
    output_dir: str = "results/financebench_single_cluster/llm/gpt54/one_shot/rule_run_individual",
) -> dict:
    """Apply a named rule to a document, retrieve matching spans, and call the LLM to answer."""

    # Step 1 — Load and apply the rule
    rule_file = Path(rules_dir) / question_slug / f"{rule_name}.py"
    rule_fn = _load_rule_fn(rule_file)
    matching_spans: list[dict] = rule_fn(document)

    # Step 2 — Sort spans in reading order and concatenate text
    texts: list[dict] = document.get("texts", [])
    text_positions: dict[int, int] = {id(s): i for i, s in enumerate(texts)}

    def _sort_key(span: dict) -> tuple:
        page = span.get("page_no", 0)
        structure = span.get("structure") or {}
        level_index = structure.get("level_index")
        if level_index is None:
            level_index = text_positions.get(id(span), 0)
        return (page, level_index)

    sorted_spans = sorted(matching_spans, key=_sort_key)
    retrieved_text = "\n\n".join(s["text"] for s in sorted_spans) if sorted_spans else ""

    # Step 3 — Count tokens in retrieved text
    retrieved_token_count = _count_tokens(retrieved_text)

    # Step 4 — Call LLM
    model_mod = importlib.import_module(f"models.{model_name}")

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

    # Step 5 — Build result record and merge into output file
    result: dict[str, Any] = {
        "rule_name": rule_name,
        "question_slug": question_slug,
        "question": question,
        "doc_name": document.get("doc_name", ""),
        "strategy": "individual",
        "predicted_answer": predicted_answer,
        "latency_seconds": round(latency_seconds, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "retrieved_token_count": retrieved_token_count,
        "retrieved_spans": sorted_spans,
        "retrieved_text": retrieved_text,
    }

    out_path = Path(output_dir) / question_slug / f"{rule_name}_individual.json"
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

    # Folder names include a doc-count suffix: {question_slug}_{n}
    rule_dirs = sorted(_glob.glob(str(_ROOT / "rules/financebench_single_cluster/llm/gpt54/one_shot" / f"{question_slug}*")))
    if not rule_dirs:
        print(f"No rule folders found matching rules/financebench_single_cluster/llm/gpt54/one_shot/{question_slug}*/")
        print("Run rule_gen_llm_coarse first to generate rules.")
        sys.exit(1)
    # Pick the folder with the most docs (largest suffix number); use its name as the slug
    rule_dir = rule_dirs[-1]
    folder_slug = Path(rule_dir).name
    rule_files = sorted(_glob.glob(str(Path(rule_dir) / "*.py")))
    if not rule_files:
        print(f"No .py files found in {rule_dir}")
        sys.exit(1)

    rule_name = Path(rule_files[0]).stem

    result = rule_apply_individual(
        document=document,
        rule_name=rule_name,
        question_slug=folder_slug,
        question=first_question,
    )

    print(f"predicted_answer:      {result['predicted_answer']}")
    print(f"retrieved_token_count: {result['retrieved_token_count']}")
