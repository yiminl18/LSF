"""Rule generation via LLM (coarse pass).

Given a collection of structurally-similar documents, a question, and ground
truth answers, asks an LLM to produce Python rule functions that locate the
answer span in any new document from the same collection.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models import gpt54 as _gpt

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a document rule engineer. Your task is to analyze how a specific question
is answered across a collection of structurally similar documents, and generate
Python rules that can reliably locate the answer in any new document from the
same collection.

Each document is represented as a JSON object with a "texts" array. Each element
in "texts" is a span with these fields:
  - text: the content (tables are in Markdown pipe format)
  - label: one of "text", "section_header", "table", "list_item"
  - page_no: integer page number
  - bold: 1 if bold, 0 otherwise
  - size: font size in points
  - structure.level: heading level — "H1", "H2", "H3", "H4", or "Body"
  - structure.path_text: breadcrumb of ancestor section headers
  - table_data.cells: list of {row, col, text, is_column_header, is_row_header}
    (only present when label == "table")
"""

_USER_PROMPT_SUFFIX = """\

---

Based on the documents and answers above, generate as many Python rules as possible
that locate the answer span(s) in any new document from this collection.

For each rule, consider the following signal types as hints:

1. PHYSICAL LOCATION — Which page(s) does the answer consistently appear on?
   Example: "answer is always on page 1 or 2"

2. SEMANTIC LOCATION — Which section header is the answer under?
   Use structure.path_text or nearby section_header spans.
   Example: "answer is under a span whose path_text contains 'Item 1'"

3. KEYWORD PROXIMITY — What keywords appear near the answer?
   Example: "answer is in a span whose text contains 'Employer Identification'"

4. DATA FEATURE — Is the answer in a table, and if so, what is the table about?
   Use label == "table" and table_data.cells.
   Example: "answer is in a table cell in row where col 0 text == 'Net Sales'"

5. TYPOGRAPHY — Is the answer in a bold span, large font, or all-caps heading?
   Example: "answer is in the first bold, all_cap span on page 1"

6. STRUCTURAL POSITION — What is the heading level or depth of the span?
   Example: "answer is in an H1 span near the top of the document"

7. ANY OTHER RULE TYPE you observe that is not listed above. Think carefully
   about patterns in the data — label combinations, sibling relationships,
   page ranges, table column/row header patterns, list item positions, etc.
   Be creative and exhaustive.

REQUIREMENTS FOR EACH RULE:
- Give the rule a short descriptive name (snake_case)
- Write it as a Python function with this exact signature:
    def rule_<name>(doc: dict) -> list[dict]:
        \"\"\"One-line description of what this rule matches.\"\"\"
        ...
        return [span, ...]   # list of matching spans from doc["texts"]
- The function must be self-contained (import json/re inside if needed)
- Return an empty list if no match is found — never raise exceptions
- Aim for high recall: prefer returning a few extra spans over missing the answer
- Generate as many rules as possible — cover every pattern you observe
- Rules may overlap; that is fine
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_question_slug(question: str) -> str:
    slug = question.lower()
    slug = re.sub(r"[^\w\s]", "", slug)
    slug = re.sub(r"\s+", "_", slug)
    return slug[:60]


def _build_user_prompt(
    documents: list[dict],
    question: str,
    ground_truth: dict,
) -> str:
    n = len(documents)
    parts: list[str] = [
        f"I have a collection of {n} financial documents that are structurally similar\n"
        "(all generated from the same SEC filing template). I want to find rules that\n"
        "describe WHERE the answer to the following question is located across all documents.\n"
        f"\nQUESTION: {question}\n"
        "\nHere are the documents with their ground truth answers:\n",
    ]
    for doc in documents:
        doc_name = doc.get("doc_name", "unknown")
        filename = doc.get("origin", {}).get("filename", doc_name + ".pdf")
        answer = ground_truth.get(filename, ground_truth.get(doc_name, "N/A"))
        spans_json = json.dumps(doc["texts"][:80], indent=2)
        parts.append(
            f"\n--- Document: {doc_name} ---\n"
            f"Answer: {answer}\n"
            f"Document JSON (texts array, first 80 spans shown):\n{spans_json}\n"
        )
    parts.append(_USER_PROMPT_SUFFIX)
    return "".join(parts)


def _extract_functions(llm_text: str) -> list[tuple[str, str, str]]:
    """Return (func_name, description, source) triples parsed from LLM output."""
    # Strip markdown code fences
    text = re.sub(r"```(?:python)?\n?", "", llm_text)
    text = re.sub(r"```\n?", "", text)

    # Split at every top-level "def rule_" to separate individual functions
    chunks = re.split(r"(?=^def rule_)", text, flags=re.MULTILINE)

    results: list[tuple[str, str, str]] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk.startswith("def rule_"):
            continue
        name_match = re.match(r"def\s+(rule_\w+)\s*\(", chunk)
        if not name_match:
            continue
        func_name = name_match.group(1)
        # Prefer triple-double-quote docstring; fall back to triple-single
        doc_match = re.search(r'"""(.*?)"""', chunk, re.DOTALL)
        if not doc_match:
            doc_match = re.search(r"'''(.*?)'''", chunk, re.DOTALL)
        description = doc_match.group(1).strip() if doc_match else ""
        results.append((func_name, description, chunk))
    return results


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def rule_gen_llm_coarse(
    documents: list[dict],
    question: str,
    ground_truth: dict,
    model_name: str = "gpt54",
    output_dir: str = "results/financebench/lsf/single_cluster/llm/gpt54/one_shot/rule_gen",
    rules_dir: str = "rules/financebench/lsf/single_cluster/llm/gpt54/one_shot",
) -> dict:
    """
    Given a collection of similar documents, a question, and ground truth answers,
    use an LLM to generate Python rules that locate the answer span in any document.

    Returns a summary dict with rule names, file paths, and run metadata.
    """
    question_slug = _make_question_slug(question)
    now = datetime.now(timezone.utc)
    timestamp_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    timestamp_file = now.strftime("%Y%m%d_%H%M%S")

    user_prompt = _build_user_prompt(documents, question, ground_truth)

    # LLM call — use the raw client so we get usage metadata
    t0 = time.monotonic()
    response = _gpt.client.chat.completions.create(
        model=_gpt.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=16000,
        temperature=0.0,
    )
    latency_seconds = time.monotonic() - t0

    llm_text = (response.choices[0].message.content or "").strip()
    usage = response.usage
    input_tokens: int = usage.prompt_tokens if usage else 0
    output_tokens: int = usage.completion_tokens if usage else 0

    # Parse rule functions from LLM output
    parsed = _extract_functions(llm_text)

    # Prepare output directories
    rule_subdir = Path(rules_dir) / f"{question_slug}_{len(documents)}_llm"
    rule_subdir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rules_list: list[dict[str, str]] = []

    if not parsed:
        # Save raw LLM text for manual inspection
        raw_path = out_dir / f"{question_slug}_{timestamp_file}_raw.txt"
        raw_path.write_text(llm_text, encoding="utf-8")
    else:
        for func_name, description, source in parsed:
            rule_file = rule_subdir / f"{func_name}.py"
            rule_file.write_text(source + "\n", encoding="utf-8")
            rules_list.append({
                "rule_name": func_name,
                "description": description,
                "file": str(rule_file),
            })

    doc_names = [d.get("doc_name", d.get("origin", {}).get("filename", "unknown")) for d in documents]

    result: dict[str, Any] = {
        "question": question,
        "question_slug": question_slug,
        "timestamp": timestamp_iso,
        "model": model_name,
        "num_documents": len(documents),
        "doc_names": doc_names,
        "latency_seconds": round(latency_seconds, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "rules": rules_list,
    }

    result_file = out_dir / f"{question_slug}_{timestamp_file}.json"
    result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result


# ---------------------------------------------------------------------------
# Quick test entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.chdir(_ROOT)

    _DATA = _ROOT / "data" / "financebench"

    # First 3 documents from sample.txt that have a reconstructed JSON
    sample_lines = (_DATA / "sample.txt").read_text(encoding="utf-8").splitlines()
    sample_pdfs = [ln.strip() for ln in sample_lines if ln.strip()]

    docs: list[dict] = []
    for pdf_name in sample_pdfs:
        stem = Path(pdf_name).stem
        json_path = _DATA / "processing" / f"{stem}_reconstructed.json"
        if json_path.exists():
            docs.append(json.loads(json_path.read_text(encoding="utf-8")))
            if len(docs) == 3:
                break

    if not docs:
        print("ERROR: no documents loaded — cannot proceed")
        sys.exit(1)

    # First question from queries.txt
    queries = (_DATA / "queries.txt").read_text(encoding="utf-8").splitlines()
    question = next(q.strip() for q in queries if q.strip())

    # Ground truth: {pdf_filename: answer_for_this_question}
    labels: dict[str, dict] = json.loads(
        (_DATA / "sample_labels.json").read_text(encoding="utf-8")
    )
    gt: dict[str, Any] = {
        pdf: data[question]
        for pdf, data in labels.items()
        if question in data
    }

    print(f"Question   : {question}")
    print(f"Documents  : {[d['doc_name'] for d in docs]}")
    print(f"GT entries : {len(gt)}")
    print("Calling rule_gen_llm_coarse …")

    result = rule_gen_llm_coarse(docs, question, gt)

    print("\nResult summary:")
    print(f"  question_slug : {result['question_slug']}")
    print(f"  model         : {result['model']}")
    print(f"  num_documents : {result['num_documents']}")
    print(f"  latency       : {result['latency_seconds']}s")
    print(f"  input_tokens  : {result['input_tokens']}")
    print(f"  output_tokens : {result['output_tokens']}")
    print(f"  rules         : {len(result['rules'])}")
    for r in result["rules"]:
        print(f"    - {r['rule_name']}: {r['description']}")
