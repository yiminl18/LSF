"""Generate ground-truth labels for court docs using gpt54.

Processes ALL queries in queries.json against every doc in data/court/text/,
one (query, doc) pair per API call. Saves incrementally.

Output:
  all_labels.json   : { "doc.pdf": { "query_text": answer, ... } }
  gen_metadata.json : per-(doc, query) tokens and latency
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import importlib
model_mod = importlib.import_module("models.gpt54")

# ── Paths ─────────────────────────────────────────────────────────────────────
QUERIES_FILE  = Path(__file__).parent / "queries.json"
TEXT_DIR      = Path(__file__).parent / "text"
LABELS_FILE   = Path(__file__).parent / "all_labels.json"
METADATA_FILE = Path(__file__).parent / "gen_metadata.json"

# ── Answer-type hints ─────────────────────────────────────────────────────────
_TYPE_HINTS = {
    "list[string]": (
        'Return a JSON array of strings, e.g. ["value1", "value2"]. '
        "If not found, return []. No explanation."
    ),
    "list of strings": (
        'Return a JSON array of strings, e.g. ["value1", "value2"]. '
        "If not found, return []. No explanation."
    ),
    "string": (
        'Return a single plain string. If not found, return "NOT FOUND". No explanation.'
    ),
    "date": (
        "Return the date as it appears in the document (e.g. \"January 15, 2025\"). "
        'If not found, return "NOT FOUND". No explanation.'
    ),
    "integer": (
        "Return a single integer. If not found, return 0. No explanation."
    ),
}

_SYSTEM = """\
You are a legal document information-extraction assistant.
Given a court document and a question, extract the answer directly from the document text.
Do not infer or guess — only use what is explicitly stated.
{type_hint}"""


def type_hint(answer_type: str) -> str:
    return _TYPE_HINTS.get(answer_type, _TYPE_HINTS["string"])


def parse_answer(raw: str, answer_type: str):
    raw = raw.strip()
    if answer_type in ("list[string]", "list of strings"):
        try:
            val = json.loads(raw)
            if isinstance(val, list):
                return val
        except Exception:
            pass
        return [raw] if raw and raw.upper() not in ("NOT FOUND", "[]", "NONE") else []
    if answer_type == "integer":
        try:
            return int(raw.replace(",", ""))
        except Exception:
            return raw
    return raw


def call_gpt54(document_text: str, question: str, answer_type: str) -> dict:
    hint = type_hint(answer_type)
    system = _SYSTEM.format(type_hint=hint)
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


# ── Load queries and docs ─────────────────────────────────────────────────────
queries   = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
txt_files = sorted(TEXT_DIR.glob("*.txt"))

print(f"Queries  : {len(queries)}")
print(f"Documents: {len(txt_files)}")
print(f"Total calls: {len(queries) * len(txt_files)}\n")

# ── Load existing outputs (resume-safe) ───────────────────────────────────────
labels:   dict = json.loads(LABELS_FILE.read_text())   if LABELS_FILE.exists()   else {}
metadata: dict = json.loads(METADATA_FILE.read_text()) if METADATA_FILE.exists() else {}

# ── Main loop: query × doc ────────────────────────────────────────────────────
total_in = total_out = total_calls = 0

for q in queries:
    question    = q["text"]
    answer_type = q["answer_type"]
    print(f"\n=== {question[:80]} ===")

    for txt_path in txt_files:
        doc_key  = txt_path.stem + ".pdf"
        meta_key = f"{doc_key}||{question}"

        if doc_key in labels and question in labels[doc_key]:
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

        if doc_key not in labels:
            labels[doc_key] = {}
        labels[doc_key][question] = result["answer"]

        metadata[meta_key] = {
            "doc":        doc_key,
            "question":   question,
            "raw":        result["raw"],
            "in_tokens":  result["in_tokens"],
            "out_tokens": result["out_tokens"],
            "latency_s":  result["latency_s"],
        }

        # Incremental save
        LABELS_FILE.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
        METADATA_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        total_in    += result["in_tokens"]
        total_out   += result["out_tokens"]
        total_calls += 1

        ans_str = json.dumps(result["answer"])[:55]
        print(f"  {doc_key:<60}  in={result['in_tokens']:>5}  out={result['out_tokens']:>3}  "
              f"lat={result['latency_s']:>4.1f}s  ans={ans_str}", flush=True)

# ── Final summary ─────────────────────────────────────────────────────────────
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
