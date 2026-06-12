#!/usr/bin/env python3
"""Run Codex QA (Baseline 1) on exactly the fallback (question, doc) pairs of a
given officeqa pipeline combo — the "fallback -> codex QA" hybrid experiment.

For the chosen combo's apply results, collect every (question_slug, doc) pair
with used_fallback=True (across sampled + unsampled), then run agentic Codex QA
on each with the requested model (default gpt54mini), judged by gpt54 — the same
run_qa + judge path as src/baseline/run_eval_officeqa.py. Reuses any pair already
present in --reuse-dir or the output dir (skip-existing).

Output: baseline_results/officeqa/<out_name>/<slug>/<doc>.json  (Baseline-1 schema)

Env required (codex CLI): PATH includes ~/.npm-global/bin and
AZURE_OPENAI_API_KEY set to the cloudbank key for the QA model.
The gpt54 judge reads credentials from local/azure.json (Python client).
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

QUERIES_FILE = _ROOT / "data/officeqa/queries.json"
TEXT_DIR = _ROOT / "data/officeqa/text"

# Reuse the judge from the officeqa baseline runner (gpt54 equivalence judge).
_runner = importlib.import_module("baseline.run_eval_officeqa")
_judge = _runner._judge
_make_slug = _runner._make_slug


def collect_fallback_pairs(apply_dir: Path) -> list[dict]:
    """Return [{slug, question, doc_name}] for every used_fallback pair."""
    pairs: list[dict] = []
    seen = set()
    for f in sorted(apply_dir.glob("*_sampled.json")) + sorted(apply_dir.glob("*_unsampled.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        slug = d["question_slug"]
        question = d["question"]
        for e in d.get("per_doc", []):
            if e.get("used_fallback"):
                key = (slug, e["doc_name"])
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({"slug": slug, "question": question, "doc_name": e["doc_name"]})
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply-dir", default="results/officeqa/grid/apply/fps/agent_codex_gpt54/agentic_codex_gpt54/default")
    ap.add_argument("--baseline", default="agentic_codex_qa_txt")
    ap.add_argument("--model", default="gpt54mini")
    ap.add_argument("--out-name", default="agentic_codex_qa_gpt54mini_fb_fps_agentcodex_agenticcodex")
    ap.add_argument("--reuse-dir", default="baseline_results/officeqa/agentic_codex_qa_gpt54mini/latest_50",
                    help="existing baseline dir to reuse matching pairs from (skip rerun)")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    baseline_mod = importlib.import_module(f"baseline.{args.baseline}")
    gpt54_mod = importlib.import_module("models.gpt54")

    apply_dir = _ROOT / args.apply_dir
    out_base = _ROOT / "baseline_results" / "officeqa" / args.out_name
    out_base.mkdir(parents=True, exist_ok=True)
    reuse_dir = _ROOT / args.reuse_dir if args.reuse_dir else None

    pairs = collect_fallback_pairs(apply_dir)
    print(f"fallback pairs: {len(pairs)}  model={args.model}  jobs={args.jobs}", flush=True)
    print(f"out={out_base}\n", flush=True)

    def process(p: dict) -> tuple[str, bool | None]:
        slug, question, doc = p["slug"], p["question"], p["doc_name"]
        q_dir = out_base / slug
        q_dir.mkdir(exist_ok=True)
        out_file = q_dir / f"{doc}.json"
        if out_file.exists():
            return ("skip-out", None)
        # reuse from prior baseline run if present
        if reuse_dir is not None:
            cand = reuse_dir / slug / f"{doc}.json"
            if cand.exists():
                out_file.write_text(cand.read_text(encoding="utf-8"), encoding="utf-8")
                return ("reused", None)
        doc_path = TEXT_DIR / f"{doc}.txt"
        if not doc_path.exists():
            return ("no-txt", None)
        try:
            result = baseline_mod.run_qa(doc_path=doc_path, question=question, model=args.model,
                                         timeout=args.timeout, log_dir=q_dir / "logs", log_stem=doc)
        except Exception as e:  # noqa: BLE001
            result = {"status": "error", "answer": None, "input_tokens": 0, "output_tokens": 0,
                      "latency_seconds": 0.0, "model": args.model, "error_message": str(e)}
        # ground truth from the apply file is not carried; judge needs GT -> load labels
        gt = _LABELS.get(doc + ".pdf", {}).get(question)
        correct = _judge(question, gt, result.get("answer"), gpt54_mod)
        record = {
            "doc_name": doc, "question": question, "question_slug": slug, "split": "fallback",
            "ground_truth": gt, "answer": result.get("answer"), "correct": correct,
            "status": result.get("status"), "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0), "latency_seconds": result.get("latency_seconds", 0.0),
            "model": result.get("model", args.model),
        }
        for k, v in result.items():
            if k not in record and k != "answer":
                record[k] = v
        out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return ("ran", correct)

    done = ran = reused = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(process, p): p for p in pairs}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                kind, _ = fut.result()
            except Exception as e:  # noqa: BLE001
                kind = f"exc:{e}"
            done += 1
            if kind == "ran":
                ran += 1
            elif kind == "reused":
                reused += 1
            if done % 25 == 0 or done == len(pairs):
                print(f"  [{done}/{len(pairs)}] ran={ran} reused={reused}  last={p['slug'][:30]}/{p['doc_name']} ({kind})", flush=True)
    print(f"\nDONE: {done} pairs  ran={ran}  reused={reused}  out={out_base}", flush=True)


# labels loaded once for the judge ground truth
_LABELS = json.loads((_ROOT / "data/officeqa/all_labels.json").read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
