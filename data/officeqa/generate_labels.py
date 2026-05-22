"""Generate ground-truth labels for OfficeQA docs using gpt54.

All 24 queries are merged into a single API call per doc.
Only processes docs listed in sample_docs.txt.

Output:
  all_labels.json   : { "doc.pdf": { "query_text": answer, ... } }
  gen_metadata.json : per-doc tokens and latency
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
QUERIES_FILE   = Path(__file__).parent / "queries.json"
TEXT_DIR       = Path(__file__).parent / "text"
SAMPLE_FILE    = Path(__file__).parent / "sample_docs.txt"
LABELS_FILE    = Path(__file__).parent / "all_labels.json"
METADATA_FILE  = Path(__file__).parent / "gen_metadata.json"

_SYSTEM = """\
You are a Treasury Bulletin information-extraction assistant.
Given a document and a list of questions, extract each answer directly
from the document text. Do not infer or guess — only use what is explicitly stated.

Return a single JSON object where:
- each KEY is the EXACT question text (copy it verbatim, no numbering prefix)
- each VALUE is the extracted answer

Type rules:
- string: plain string, or "NOT FOUND" if absent
- float: numeric float (digits only, no units), or null if absent
- integer: integer (digits only), or null if absent
- categorical: exact categorical value from the document, or "NOT FOUND" if absent

Output only the JSON object, no explanation."""


def build_prompt(document_text: str, queries: list[dict]) -> str:
    lines = ["Document:", document_text, "", "Questions:"]
    for q in queries:
        lines.append(f"- {q['text']}  [type: {q['answer_type']}]")
    return "\n".join(lines)


def parse_response(raw: str, queries: list[dict]) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            # strip " [type: ...]" suffix from keys if model included it
            import re
            clean = {}
            for k, v in obj.items():
                clean_k = re.sub(r'\s+\[type:[^\]]*\]$', '', k).strip()
                clean[clean_k] = v
            return clean
    except Exception:
        pass
    return {q["text"]: raw for q in queries}


def call_gpt54_batch(document_text: str, queries: list[dict]) -> dict:
    user = build_prompt(document_text, queries)
    t0 = time.time()
    resp = model_mod.client.chat.completions.create(
        model=model_mod.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": user},
        ],
        max_completion_tokens=1024,
        temperature=0.0,
    )
    latency = round(time.time() - t0, 3)
    raw = (resp.choices[0].message.content or "").strip()
    answers = parse_response(raw, queries)
    return {
        "answers":    answers,
        "raw":        raw,
        "in_tokens":  resp.usage.prompt_tokens     if resp.usage else 0,
        "out_tokens": resp.usage.completion_tokens if resp.usage else 0,
        "latency_s":  latency,
    }


# ── Load queries and sampled doc list ─────────────────────────────────────────
queries    = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
sample_txt = [l.strip() for l in SAMPLE_FILE.read_text().splitlines() if l.strip()]
txt_files  = [TEXT_DIR / name for name in sample_txt if (TEXT_DIR / name).exists()]

print(f"Queries  : {len(queries)}")
print(f"Documents: {len(txt_files)}")
print(f"Total calls: {len(txt_files)} (one per doc, all queries merged)\n")

# ── Load existing outputs (resume-safe) ───────────────────────────────────────
labels:   dict = json.loads(LABELS_FILE.read_text())   if LABELS_FILE.exists()   else {}
metadata: dict = json.loads(METADATA_FILE.read_text()) if METADATA_FILE.exists() else {}

# ── Main loop: one call per doc ───────────────────────────────────────────────
for i, txt_path in enumerate(txt_files, 1):
    doc_key = txt_path.stem + ".pdf"

    if doc_key in labels and len(labels[doc_key]) == len(queries):
        print(f"[{i}/{len(txt_files)}] Skip (done): {doc_key}", flush=True)
        continue

    text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) < 50:
        print(f"[{i}/{len(txt_files)}] SKIP (too short): {doc_key}", flush=True)
        continue

    try:
        result = call_gpt54_batch(text, queries)
    except Exception as e:
        print(f"[{i}/{len(txt_files)}] ERR {doc_key}: {e}", flush=True)
        continue

    labels[doc_key] = result["answers"]
    metadata[doc_key] = {
        "doc":        doc_key,
        "in_tokens":  result["in_tokens"],
        "out_tokens": result["out_tokens"],
        "latency_s":  result["latency_s"],
        "raw":        result["raw"],
    }

    LABELS_FILE.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
    METADATA_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[{i}/{len(txt_files)}] {doc_key:<50}  in={result['in_tokens']:>6}  "
          f"out={result['out_tokens']:>4}  lat={result['latency_s']:>5.1f}s", flush=True)

# ── Final summary ─────────────────────────────────────────────────────────────
all_meta = list(metadata.values())
if all_meta:
    ti = sum(r["in_tokens"]  for r in all_meta)
    to = sum(r["out_tokens"] for r in all_meta)
    tl = sum(r["latency_s"]  for r in all_meta)
    print(f"\n{'='*70}")
    print(f"Total calls : {len(all_meta)}")
    print(f"Input tokens: {ti:,}  (~${ti/1e6*2.5:.2f})")
    print(f"Output tokens: {to:,}  (~${to/1e6*10:.2f})")
    print(f"Total latency: {tl:.0f}s ({tl/60:.1f}m)")
    print(f"Labels  : {LABELS_FILE}")
    print(f"Metadata: {METADATA_FILE}")
