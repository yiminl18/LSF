"""Generate ground-truth labels for PRODUCT (EMA EPAR) docs using gpt54.

Mimics data/nopv/generate_labels.py: ALL queries in queries.json against the
docs listed in sampled_docs.txt (the 200-doc sample), one (query, doc) pair per
API call. Saves incrementally and is resume-safe.

Usage:
  python3 data/product/generate_labels.py            # all docs in manifest
  python3 data/product/generate_labels.py --limit 10 # smoke test: first 10 docs

Output:
  all_labels.json   : { "doc.pdf": { "query_text": answer, ... } }
  gen_metadata.json : per-(doc, query) tokens and latency
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import importlib
model_mod = importlib.import_module("models.gpt54")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE          = Path(__file__).parent
QUERIES_FILE  = HERE / "queries.json"
TEXT_DIR      = HERE / "text"
MANIFEST_FILE = HERE / "sampled_docs.txt"
LABELS_FILE   = HERE / "all_labels.json"
METADATA_FILE = HERE / "gen_metadata.json"

# ── Answer-type hints ─────────────────────────────────────────────────────────
_TYPE_HINTS = {
    "string": (
        'Return a single plain string. If not found, return "NOT FOUND". No explanation.'
    ),
    "date": (
        "Return the date as it appears in the document (e.g. \"15 March 2024\"). "
        'If not found, return "NOT FOUND". No explanation.'
    ),
    "boolean": (
        'Return exactly "Yes" or "No". If it cannot be determined from the document, '
        'return "NOT FOUND". No explanation.'
    ),
    "number": (
        "Return a single number as written in the document (digits, may include a "
        'decimal point or unit). If not found, return "NOT FOUND". No explanation.'
    ),
}

_SYSTEM = """\
You are a pharmaceutical product-information extraction assistant.
Given an EMA EPAR product-information (SmPC) document and a question,
extract the answer directly from the document text.
Do not infer or guess — only use what is explicitly stated.
{type_hint}"""


def type_hint(answer_type: str) -> str:
    return _TYPE_HINTS.get(answer_type, _TYPE_HINTS["string"])


def parse_answer(raw: str, answer_type: str):
    raw = raw.strip()
    if answer_type == "boolean":
        low = raw.lower()
        if low.startswith("yes"):
            return True
        if low.startswith("no"):
            return False
        return raw
    if answer_type == "number":
        try:
            return float(raw.replace(",", "")) if "." in raw else int(raw.replace(",", ""))
        except Exception:
            return raw
    return raw


def call_gpt54(document_text: str, question: str, answer_type: str) -> dict:
    system = _SYSTEM.format(type_hint=type_hint(answer_type))
    user = f"Document:\n{document_text}\n\nQuestion: {question}"

    t0 = time.time()
    resp = model_mod.client.chat.completions.create(
        model=model_mod.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_completion_tokens=512,
        temperature=0.0,
    )
    latency = round(time.time() - t0, 3)

    raw = (resp.choices[0].message.content or "").strip()
    return {
        "answer":     parse_answer(raw, answer_type),
        "raw":        raw,
        "in_tokens":  resp.usage.prompt_tokens     if resp.usage else 0,
        "out_tokens": resp.usage.completion_tokens if resp.usage else 0,
        "latency_s":  latency,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N docs of the manifest (0 = all)")
    args = ap.parse_args()

    queries = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
    manifest = [ln.strip() for ln in MANIFEST_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.limit:
        manifest = manifest[:args.limit]
    txt_files = [TEXT_DIR / name for name in manifest]

    print(f"Queries  : {len(queries)}")
    print(f"Documents: {len(txt_files)} (from {MANIFEST_FILE.name}{', limited' if args.limit else ''})")
    print(f"Total calls: {len(queries) * len(txt_files)}\n")

    labels:   dict = json.loads(LABELS_FILE.read_text())   if LABELS_FILE.exists()   else {}
    metadata: dict = json.loads(METADATA_FILE.read_text()) if METADATA_FILE.exists() else {}

    for q in queries:
        question    = q["text"]
        answer_type = q["answer_type"]
        print(f"\n=== [{answer_type}] {question[:75]} ===")

        for txt_path in txt_files:
            doc_key  = txt_path.stem + ".pdf"
            meta_key = f"{doc_key}||{question}"

            if doc_key in labels and question in labels[doc_key]:
                continue
            if not txt_path.exists():
                print(f"  MISS (no text): {doc_key}", flush=True)
                continue

            text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
            if len(text) < 50:
                print(f"  SKIP (too short): {doc_key}", flush=True)
                continue

            try:
                result = call_gpt54(text, question, answer_type)
            except Exception as e:
                print(f"  ERR {doc_key}: {e}", flush=True)
                continue

            labels.setdefault(doc_key, {})[question] = result["answer"]
            metadata[meta_key] = {
                "doc":        doc_key,
                "question":   question,
                "raw":        result["raw"],
                "in_tokens":  result["in_tokens"],
                "out_tokens": result["out_tokens"],
                "latency_s":  result["latency_s"],
            }

            LABELS_FILE.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
            METADATA_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

            ans_str = json.dumps(result["answer"], ensure_ascii=False)[:55]
            print(f"  {doc_key:<55}  in={result['in_tokens']:>6}  out={result['out_tokens']:>3}  "
                  f"lat={result['latency_s']:>4.1f}s  ans={ans_str}", flush=True)

    all_meta = list(metadata.values())
    if all_meta:
        ti = sum(r["in_tokens"]  for r in all_meta)
        to = sum(r["out_tokens"] for r in all_meta)
        tl = sum(r["latency_s"]  for r in all_meta)
        print(f"\n{'='*70}")
        print(f"Total calls : {len(all_meta)}")
        print(f"Input tokens: {ti:,}  (~${ti/1e6*2.5:.2f})")
        print(f"Output tokens:{to:,}  (~${to/1e6*15:.2f})")
        print(f"Total latency: {tl:.0f}s ({tl/60:.1f}m)")
        print(f"Labels  : {LABELS_FILE}")
        print(f"Metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()
