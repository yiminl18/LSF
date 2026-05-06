"""Task prompt builder for rule generation, to be passed to `claude -p`."""

from __future__ import annotations

import subprocess
from pathlib import Path


TASK_PROMPT = """\
You are working inside the LSF project root. Your task is to generate a minimal \
set of Python span-retrieval rules for the following question over SEC financial filings.

QUESTION: {question}

INPUTS
  Labels file (ground truth):  {labels_file}
  Document JSON files:         {processing_dir}/<DOC_NAME>_reconstructed.json
  Existing rules to check:     {rules_dir}

---

DEFINITIONS

Cost (per document)
  Apply the union of all rules to a document. Concatenate the text of every \
returned span. Compute:
    cost = tiktoken_tokens(retrieved_text) / tiktoken_tokens(full_document_text)
  where full_document_text is the concatenation of all spans in doc["texts"].
  avg_cost = mean of cost over all sampled documents.
  A low avg_cost means the rules are selective — they retrieve only the \
relevant fragment instead of large swaths of the document.

Merge accuracy
  For each sampled document, apply every rule, take the union of all returned \
spans (deduplicated by identity), concatenate their text, and pass it to the LLM \
with the question. The document is "correct" if the LLM's answer matches the \
ground truth (judged by a second LLM call for semantic equivalence).
  merge_accuracy = (number of correct documents) / (total sampled documents)
  During iterative development you may use a fast proxy: a document is "hit" if \
the ground truth string appears as a substring (case-insensitive) in the \
retrieved text. Use this proxy to iterate cheaply; call the LLM judge only \
for final verification.

---

OBJECTIVES (in priority order)

1. merge_accuracy >= 0.95 on the sampled documents.
2. avg_cost as small as possible. Target avg_cost < 0.05 (retrieved tokens are \
less than 5% of the full document). After hitting the accuracy threshold, tighten \
high-cost rules without dropping accuracy.
3. Fewest rules possible. Prefer one broad rule that covers 8 of 10 documents \
over three narrow rules that each cover 3. A rule covering fewer than 2 documents \
should be merged into a broader rule or dropped.

---

DOCUMENT STRUCTURE

Each document JSON has a "texts" list of span dicts with these fields:
  text            — string content; tables are in Markdown pipe format
  label           — "text" | "section_header" | "table" | "list_item"
  page_no         — integer page number
  bold            — 1 if bold, 0 otherwise
  size            — font size in points
  structure.level — "H1" | "H2" | "H3" | "H4" | "Body"
  structure.path_text — pipe-separated breadcrumb of ancestor section headers
  table_data.cells — list of {{row, col, text, is_column_header, is_row_header}}
                     (only present when label == "table")

RULE INTERFACE

Each rule is a standalone Python function saved as a .py file:

  def rule_<descriptive_name>(doc: dict) -> list[dict]:
      """One-line description of what this rule targets."""
      try:
          return [span for span in doc["texts"] if <condition>]
      except Exception:
          return []

Rules must be self-contained (all imports inside the function body), must never \
raise, and must return a list of span dicts from doc["texts"].

---

WORKFLOW

1. Read existing rules in {rules_dir} first. Understand which structural \
   patterns they exploit (page number, section path, label, font size, keywords \
   in table cells). Do not duplicate patterns that already work well.

2. Load the sampled documents and ground truth answers from {labels_file}. \
   The labels file maps "DOCNAME.pdf" to a dict of question → answer. Extract \
   the answers for this specific question to get your ground_truth dict.

3. Study the documents. For each sampled doc, locate the span containing the \
   ground truth answer and note its structural fields (page_no, label, level, \
   bold, path_text). Look for the dominant pattern shared by most documents — \
   that pattern becomes your first, broadest rule.

4. Write the broadest rule first. Test it: compute the hit rate (substring match) \
   and avg_cost across all sampled docs. If avg_cost > 0.05, tighten the filter \
   (add page_no, label, or path_text constraints) and retest.

5. Find uncovered documents — those where no current rule retrieves the answer. \
   Diagnose each by inspecting its answer span's structural fields and comparing \
   them to the passing docs. Write a targeted rule only if it covers at least \
   2 uncovered documents.

6. Repeat steps 4–5 until substring-match coverage is >= 0.95.

7. Run the final LLM-judge evaluation using src/rule_apply_merge.py and \
   src/eval_rule.py (or equivalent) to confirm merge_accuracy >= 0.95.

8. If any rule has avg_cost > 0.05, tighten it (narrower page range, stricter \
   label or path_text filter) and re-verify accuracy is preserved.

9. Save each rule as a .py file to {rules_dir}/{question_slug}/.

10. Print a final summary: number of rules, merge_accuracy, avg_cost, and for \
    each rule: its name, how many sampled docs it covers, and its individual \
    avg_cost.

---

DESIGN HINTS

- Anchor rules on structure.path_text or page_no rather than free-text keywords \
  alone — structural anchors generalise better across companies and years.
- For cover-page facts (company name, ticker, address, shares outstanding), the \
  answer is almost always on page 1 or 2. A page_no filter is cheap and precise.
- For financial figures (revenue, net income, total assets), anchor on section \
  headers (Item 8, Consolidated Statements) found via path_text, then restrict \
  to label == "table".
- If a rule returns entire sections, add a keyword filter on table_data.cells or \
  span text to narrow it down.
- Do not write a new rule just to get from 0.95 to 1.0 if doing so doubles \
  avg_cost. The cost objective matters.
"""


def build_prompt(
    question: str,
    labels_file: str = "data/financebench/sample_doc_labels.json",
    processing_dir: str = "data/financebench/processing",
    rules_dir: str = "rules/agent/financebench_agent",
) -> str:
    import re
    question_slug = re.sub(r"[^\w]", "_", question.lower())[:60].rstrip("_")
    return TASK_PROMPT.format(
        question=question,
        question_repr=repr(question),
        question_slug=question_slug,
        labels_file=labels_file,
        labels_file_repr=repr(labels_file),
        processing_dir=processing_dir,
        processing_dir_repr=repr(processing_dir),
        rules_dir=rules_dir,
        rules_dir_repr=repr(rules_dir),
    )


def run(
    question: str,
    labels_file: str = "data/financebench/sample_doc_labels.json",
    processing_dir: str = "data/financebench/processing",
    rules_dir: str = "rules/agent/financebench_agent",
    model: str = "claude-opus-4-5",
    cwd: str | None = None,
) -> str:
    prompt = build_prompt(
        question=question,
        labels_file=labels_file,
        processing_dir=processing_dir,
        rules_dir=rules_dir,
    )
    project_root = cwd or str(Path(__file__).resolve().parent)
    result = subprocess.run(
        ["claude", "--model", model, "-p", prompt],
        capture_output=True,
        text=True,
        cwd=project_root,
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude exited {result.returncode}:\n{result.stderr}")
    return result.stdout


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate rules via claude -p.")
    parser.add_argument("question")
    parser.add_argument("--labels-file",    default="data/financebench/sample_doc_labels.json")
    parser.add_argument("--processing-dir", default="data/financebench/processing")
    parser.add_argument("--rules-dir",      default="rules/agent/financebench_agent")
    parser.add_argument("--model",          default="claude-opus-4-5")
    parser.add_argument("--cwd",            default=None)
    parser.add_argument("--print-prompt",   action="store_true")
    args = parser.parse_args()

    if args.print_prompt:
        print(build_prompt(args.question, args.labels_file,
                           args.processing_dir, args.rules_dir))
    else:
        print(run(args.question, args.labels_file, args.processing_dir,
                  args.rules_dir, args.model, args.cwd))
