"""Validation-guarded rule generation (fork of `agent_codex`).

Same Codex rule-generation agent as `agent_codex.py`, plus a generalization
guard: the agent is handed a held-out VALIDATION document set (carved from the
unsampled pool, never used to design rules) and must confirm that rules trained
on the sampled docs hold up on the validation docs *before* saving — broadening
overfit rules if validation accuracy drops below sampled accuracy.

The final TEST evaluation excludes these validation docs (option-3 reporting in
`src/pipeline.py`), so the validation step here does not leak into the test
score. The legacy full-unsampled number (which *includes* the validation docs)
is also reported for apples-to-apples comparison with the existing
`agent_codex` rows.

Wiring lives in `src/pipeline.py` under `rule_gen_strategy='agent_codex_val'`.
This module deliberately reuses `agent_codex.TASK_PROMPT` verbatim and only
appends a VALIDATION GUARD section, so the base rule-generation behaviour is
identical to the baseline and the only difference under test is the guard.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# When invoked directly as a subprocess script (python3 src/rule_gen/agent_codex_val.py),
# `src/` is not on sys.path the way it is when pipeline.py imports this module, so the
# `rule_gen` package isn't importable. Bootstrap it before importing the base module.
_SRC = Path(__file__).resolve().parent.parent  # .../src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rule_gen import agent_codex as _base


# Appended verbatim after the base task prompt. Only `{val_doc_list}`,
# `{labels_file}` and `{question_slug}` are substituted (all supplied by
# build_prompt); no other single braces appear here.
VALIDATION_GUARD = '''

---

VALIDATION GUARD (generalization check — the KEY DIFFERENCE from the standard generator)

You are ALSO given a set of held-out VALIDATION documents. You did NOT design
rules on these; they exist only to verify that rules learned on the sampled
documents generalize to documents you have never optimized against.

  Validation documents (held-out):
{val_doc_list}

The ground-truth labels file ({labels_file}) contains answers for these
validation documents too (same "DOCNAME.pdf" -> question -> answer format as the
sampled docs).

Run this guard AFTER you reach merge_accuracy >= 0.95 on the SAMPLED documents
and BEFORE you save the final rules (i.e. between WORKFLOW step 7 and step 9):

V1. Apply your current rule set to every VALIDATION document. Compute
    val_accuracy using the same LLM-judge merge-accuracy definition you used on
    the sampled docs, plus val_avg_cost. (You may use the substring proxy to
    iterate cheaply, but confirm the final val_accuracy with the LLM judge.)

V2. Compare val_accuracy against your sampled merge_accuracy:
    - If val_accuracy >= sampled_accuracy - 0.05: the rules generalize. Accept
      and proceed to save (step 9).
    - If val_accuracy <  sampled_accuracy - 0.05: the rules OVERFIT the sample.
      BROADEN them — relax narrow, sample-specific constraints (hard-coded page
      numbers, exact keyword strings, document-specific path_text) toward
      structural anchors that hold across documents; merge narrow rules into
      broader ones; drop rules that only ever fire on the sampled docs. Then
      re-measure on BOTH the sampled docs (must stay >= 0.95) and the validation
      docs.

V3. Repeat the broaden-and-recheck loop at most 3 times. Keep the rule set with
    the HIGHEST val_accuracy among the versions that held sampled merge_accuracy
    >= 0.95. If no version reached 0.95 on the sample, keep the best sampled
    version.

V4. Only the kept rule set is saved (step 9). In the
    {question_slug}_rule_gen.json log, additionally record:
      "sampled_accuracy":  <float>,   // merge accuracy on the sampled docs
      "val_accuracy":      <float>,   // merge accuracy on the validation docs
      "broadening_passes": <int>      // how many times you broadened (0 if none)
    Print all three in the final summary (step 11).

Priority order is unchanged: sampled accuracy first, then generalization (a
small sample-to-validation gap), then cost. Do NOT sacrifice sampled accuracy to
chase validation accuracy.
'''


TASK_PROMPT = _base.TASK_PROMPT + VALIDATION_GUARD
_MODEL_ALIASES = _base._MODEL_ALIASES


def build_prompt(
    question: str,
    docs: list[str],                    # sampled DOC_NAMEs (no .pdf suffix)
    labels_file: str = "data/financebench/sample/single_cluster/random/sample_doc_labels.json",
    processing_dir: str = "data/financebench/processing",
    rules_dir: str = "rules/financebench/lsf/single_cluster/agent/gpt54/codex/raw",
    model: str = "gpt54",
    question_slug: str | None = None,
    val_docs: list[str] | None = None,  # held-out validation DOC_NAMEs (no .pdf)
) -> str:
    import re
    if not question_slug:
        question_slug = re.sub(r"[^\w]", "_", question.lower())[:60].rstrip("_")
    doc_list = "\n".join(f"    - {d}" for d in docs) if docs else "    (all docs in labels file)"
    val_doc_list = (
        "\n".join(f"    - {d}" for d in val_docs) if val_docs else "    (none provided)"
    )
    resolved_model = _MODEL_ALIASES.get(model, model)
    return TASK_PROMPT.format(
        question=question,
        question_slug=question_slug,
        doc_list=doc_list,
        labels_file=labels_file,
        processing_dir=processing_dir,
        rules_dir=rules_dir,
        model_name=resolved_model,
        val_doc_list=val_doc_list,
    )


def run(
    question: str,
    docs: list[str],
    labels_file: str = "data/financebench/sample/single_cluster/random/sample_doc_labels.json",
    processing_dir: str = "data/financebench/processing",
    rules_dir: str = "rules/financebench/lsf/single_cluster/agent/gpt54/codex/raw",
    model: str = "gpt54",
    cwd: str | None = None,
    timeout: int = 5400,
    output_last_message: str | None = None,
    question_slug: str | None = None,
    val_docs: list[str] | None = None,
) -> str:
    prompt = build_prompt(
        question=question,
        docs=docs,
        labels_file=labels_file,
        processing_dir=processing_dir,
        rules_dir=rules_dir,
        model=model,
        question_slug=question_slug,
        val_docs=val_docs,
    )
    resolved_model = _MODEL_ALIASES.get(model, model)
    project_root = cwd or str(Path(__file__).resolve().parents[1])

    codex_bin = shutil.which("codex")
    if not codex_bin:
        raise RuntimeError("codex CLI not found on PATH")

    cmd = [
        codex_bin,
        "--ask-for-approval", "never",
        "exec",
        "--json",
        "--color", "never",
        "--model", resolved_model,
        "--cd", project_root,
        "--sandbox", "danger-full-access",
    ]
    if output_last_message:
        cmd.extend(["--output-last-message", output_last_message])
    cmd.append(prompt)

    result = subprocess.run(
        cmd,
        input="",
        capture_output=True,
        text=True,
        cwd=project_root,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"codex exited {result.returncode}:\n{result.stderr}")
    return result.stdout


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate rules via codex exec, with a held-out validation guard.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("question", help="The QA question to generate rules for.")
    parser.add_argument("--docs", nargs="+", required=True, metavar="DOC_NAME",
                        help="Sampled document names (no .pdf suffix).")
    parser.add_argument("--val-docs", nargs="*", default=None, metavar="DOC_NAME",
                        help="Held-out validation document names (no .pdf suffix).")
    parser.add_argument("--labels-file",    default="data/financebench/sample/single_cluster/random/sample_doc_labels.json")
    parser.add_argument("--processing-dir", default="data/financebench/processing")
    parser.add_argument("--rules-dir",      default="rules/financebench/lsf/single_cluster/agent/gpt54/codex/raw")
    parser.add_argument("--model",          default="gpt54",
                        help="Model alias (gpt54/gpt54mini) or full model string.")
    parser.add_argument("--cwd",            default=None)
    parser.add_argument("--timeout",        type=int, default=5400,
                        help="Per-question subprocess timeout in seconds (default 5400 = 90m).")
    parser.add_argument("--output-last-message", default=None)
    parser.add_argument("--print-prompt",   action="store_true",
                        help="Print the prompt and exit without running codex.")
    parser.add_argument("--question-slug",  default=None,
                        help="Override the auto-derived question slug (grid uses this).")
    args = parser.parse_args()

    if args.print_prompt:
        print(build_prompt(args.question, args.docs, args.labels_file,
                           args.processing_dir, args.rules_dir, args.model,
                           question_slug=args.question_slug, val_docs=args.val_docs))
    else:
        print(run(args.question, args.docs, args.labels_file,
                  args.processing_dir, args.rules_dir, args.model,
                  args.cwd, args.timeout, args.output_last_message,
                  question_slug=args.question_slug, val_docs=args.val_docs))
