#!/usr/bin/env python3
"""Verify merge-accuracy of a rule subset on the sampled docs via gpt54 judge.

For each sampled doc d:
  1. Apply the union of `rules` to d via `rule_apply_merge` (gpt54 QA).
  2. Judge predicted vs ground_truth via `eval_judge.judge` (gpt54).
  3. Compare against `eval_merge/<slug>_sampled.json::per_doc[*].correct`
     to identify which D* docs (full-pool-correct) are missed by the subset.

Returns the per-doc verdict and the hard-constraint signal `missed_in_D_star`.
Cost: 2 LLM calls per doc (QA + judge) = ~20 calls per question with 10 sampled.

Usage:
    python tools/verify_accuracy.py --question-slug <slug> \
        --question "What is ..." --rules rule_a rule_b rule_c
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_THIS))

from rule_apply_merge              import rule_apply_merge      # noqa: E402
from rule_refinement.eval_judge    import judge                 # noqa: E402
from rule_refinement.baseline_targets import load_target_docs   # noqa: E402

from _paths import (
    RULES_BASE_DIR, SAMPLED_LABELS_FILE, PROCESSING_DIR,
    EVAL_MERGE_DIR, SELECTOR_RUN_AGENT_DIR,
)  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Verify merge accuracy of a rule subset on sampled docs.")
    ap.add_argument("--question-slug", required=True,
                    help="e.g. what_is_the_registrants_telephone_number_10_llm (the rule folder name)")
    ap.add_argument("--question", required=True, help="natural-language question text")
    ap.add_argument("--rules", nargs="+", required=True)
    ap.add_argument("--rules-dir",   default=str(RULES_BASE_DIR))
    ap.add_argument("--labels-file", default=str(SAMPLED_LABELS_FILE))
    ap.add_argument("--processing-dir", default=str(PROCESSING_DIR))
    ap.add_argument("--eval-merge-dir", default=str(EVAL_MERGE_DIR))
    ap.add_argument("--output-dir",  default=str(SELECTOR_RUN_AGENT_DIR),
                    help="where per-call rule_apply_merge intermediates land")
    ap.add_argument("--model-name",  default="gpt54")
    ap.add_argument("--format", choices=("text", "json"), default="json")
    args = ap.parse_args()

    # eval_merge file naming uses the slug WITHOUT the "_llm" suffix
    output_slug = args.question_slug.replace("_llm", "")
    eval_merge_path = Path(args.eval_merge_dir) / f"{output_slug}_sampled.json"
    if not eval_merge_path.exists():
        print(f"ERROR: no eval_merge file at {eval_merge_path}", file=sys.stderr)
        sys.exit(2)

    # D* = docs where full-pool merge solves correctly
    D_star = load_target_docs(eval_merge_path)

    # Load labels and sampled docs
    labels = json.loads(Path(args.labels_file).read_text(encoding="utf-8"))
    doc_names = [k.replace(".pdf", "") for k in labels.keys()]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    per_doc = []
    correct_S = set()
    qa_in_tot = qa_out_tot = j_in_tot = j_out_tot = 0

    for dn in doc_names:
        doc_path = Path(args.processing_dir) / f"{dn}_reconstructed.json"
        if not doc_path.exists():
            per_doc.append({"doc_name": dn, "correct_S": False, "in_D_star": dn in D_star,
                            "error": "doc file missing"})
            continue
        document = json.loads(doc_path.read_text(encoding="utf-8"))
        gt = labels.get(dn + ".pdf", {}).get(args.question)
        try:
            res = rule_apply_merge(
                document=document,
                rule_names=args.rules,
                question_slug=args.question_slug,
                question=args.question,
                model_name=args.model_name,
                rules_dir=args.rules_dir,
                output_dir=args.output_dir,
            )
            predicted = res.get("predicted_answer")
            qa_in_tot  += res.get("input_tokens", 0)
            qa_out_tot += res.get("output_tokens", 0)
        except Exception as e:
            per_doc.append({"doc_name": dn, "correct_S": False, "in_D_star": dn in D_star,
                            "error": f"rule_apply_merge: {e}"})
            continue

        try:
            ok, j_in, j_out = judge(args.question, gt, predicted, model_name=args.model_name)
            j_in_tot  += j_in
            j_out_tot += j_out
        except Exception as e:
            per_doc.append({"doc_name": dn, "correct_S": False, "in_D_star": dn in D_star,
                            "error": f"judge: {e}"})
            continue

        if ok:
            correct_S.add(dn)
        per_doc.append({
            "doc_name":   dn,
            "predicted":  predicted,
            "in_D_star":  dn in D_star,
            "correct_S":  ok,
        })

    missed_in_D_star = sorted(D_star - correct_S)
    extra_outside_D_star = sorted(correct_S - D_star)
    match_rate = len(correct_S & D_star) / len(D_star) if D_star else 1.0

    result = {
        "question_slug":         args.question_slug,
        "n_rules":               len(args.rules),
        "rules":                 list(args.rules),
        "D_star_size":           len(D_star),
        "correct_S_size":        len(correct_S),
        "match_rate":            round(match_rate, 4),
        "missed_in_D_star":      missed_in_D_star,
        "extra_outside_D_star":  extra_outside_D_star,
        "hard_constraint_satisfied": len(missed_in_D_star) == 0,
        "per_doc":               per_doc,
        "tokens": {
            "qa_input":  qa_in_tot,
            "qa_output": qa_out_tot,
            "j_input":   j_in_tot,
            "j_output":  j_out_tot,
        },
    }

    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"match_rate (on D*): {result['match_rate']}")
        print(f"missed_in_D_star: {missed_in_D_star or 'NONE'}")
        print(f"tokens: qa_in={qa_in_tot} qa_out={qa_out_tot} j_in={j_in_tot} j_out={j_out_tot}")


if __name__ == "__main__":
    main()
