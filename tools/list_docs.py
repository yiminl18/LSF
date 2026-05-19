#!/usr/bin/env python3
"""List the reconstructed JSON paths for every doc in a labels file. Free.

Used by the agentic generation pipeline so the agent knows which docs it can
inspect before authoring rules. The labels file is the source of truth for
which docs are in scope; this tool just maps each entry to its corresponding
`*_reconstructed.json` under PROCESSING_DIR.

Usage:
    python tools/list_docs.py --labels-file data/financebench/sample/single_cluster/random/sample_doc_labels.json
    python tools/list_docs.py --labels-file data/financebench/sample/single_cluster/fps/sample_doc_labels.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
from _paths import PROCESSING_DIR, SAMPLED_LABELS_FILE  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="List reconstructed-JSON paths for the docs in a labels file.")
    ap.add_argument("--labels-file", default=str(SAMPLED_LABELS_FILE))
    ap.add_argument("--processing-dir", default=str(PROCESSING_DIR))
    ap.add_argument("--format", choices=("text", "json"), default="json")
    args = ap.parse_args()

    labels = json.loads(Path(args.labels_file).read_text(encoding="utf-8"))

    docs = []
    missing = []
    for pdf_key in labels:
        stem = pdf_key.replace(".pdf", "")
        json_path = Path(args.processing_dir) / f"{stem}_reconstructed.json"
        if not json_path.exists():
            missing.append(stem)
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            spans = data.get("texts", []) or []
            pages = sorted({s.get("page_no") for s in spans if s.get("page_no") is not None})
            docs.append({
                "stem":      stem,
                "json_path": str(json_path),
                "n_spans":   len(spans),
                "n_pages":   len(pages),
                "page_min":  pages[0] if pages else None,
                "page_max":  pages[-1] if pages else None,
            })
        except Exception as e:
            docs.append({"stem": stem, "json_path": str(json_path), "error": str(e)})

    result = {
        "labels_file":    str(args.labels_file),
        "processing_dir": str(args.processing_dir),
        "n_docs":         len(docs),
        "docs":           docs,
        "missing":        missing,
    }

    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"# {len(docs)} reconstructed docs ({len(missing)} missing)")
        for d in docs:
            print(f"- {d['stem']}: spans={d.get('n_spans','?')}  pages={d.get('n_pages','?')}")
        for m in missing:
            print(f"  MISSING: {m}")


if __name__ == "__main__":
    main()
