#!/usr/bin/env python3
"""Read a reconstructed-JSON doc and return a focused view of its spans. Free.

The agentic generator uses this to inspect documents before authoring rules.
Returns a compact list of spans, optionally filtered by page or substring, with
the same fields the rule function will see at runtime (text, page_no, size,
bold, label, structure.{level, path_text}, etc.).

Usage:
    python tools/read_doc_json.py --doc AMCOR_2019_10K --page 1
    python tools/read_doc_json.py --doc AMCOR_2019_10K --pages 1-3 --filter "exact name"
    python tools/read_doc_json.py --doc AMCOR_2019_10K --max-spans 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
from _paths import PROCESSING_DIR  # noqa: E402


def _parse_pages(s: str) -> set[int] | None:
    if not s:
        return None
    out: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _compact_span(s: dict, include_structure: bool) -> dict:
    out = {
        "text":    s.get("text", ""),
        "page_no": s.get("page_no"),
        "size":    s.get("size"),
        "bold":    s.get("bold"),
        "all_cap": s.get("all_cap"),
        "label":   s.get("label"),
    }
    if include_structure:
        st = s.get("structure") or {}
        out["structure"] = {
            "level":     st.get("level"),
            "path_text": st.get("path_text"),
            "depth":     st.get("depth"),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="Read a reconstructed JSON and show a filtered view of its spans.")
    ap.add_argument("--doc", required=True, help="doc stem (no .pdf, no _reconstructed.json)")
    ap.add_argument("--processing-dir", default=str(PROCESSING_DIR))
    ap.add_argument("--page", type=int, help="single page (1-indexed)")
    ap.add_argument("--pages", help="page range/list, e.g. '1-3' or '1,3,5'")
    ap.add_argument("--filter", default="", help="case-insensitive substring filter on span text")
    ap.add_argument("--max-spans", type=int, default=80, help="hard cap on spans returned (default 80)")
    ap.add_argument("--include-structure", action="store_true", default=True,
                    help="include structure.{level, path_text, depth} (default on)")
    ap.add_argument("--format", choices=("text", "json"), default="json")
    args = ap.parse_args()

    json_path = Path(args.processing_dir) / f"{args.doc}_reconstructed.json"
    if not json_path.exists():
        # Allow passing a full path too
        alt = Path(args.doc)
        if alt.exists() and alt.suffix == ".json":
            json_path = alt
        else:
            print(f"ERROR: not found: {json_path}", file=sys.stderr)
            sys.exit(2)

    doc = json.loads(json_path.read_text(encoding="utf-8"))
    spans = doc.get("texts", []) or []

    page_filter = _parse_pages(args.pages) if args.pages else None
    if args.page is not None:
        page_filter = (page_filter or set()) | {args.page}
    needle = args.filter.strip().lower()

    selected = []
    for s in spans:
        if page_filter is not None and s.get("page_no") not in page_filter:
            continue
        if needle and needle not in (s.get("text", "") or "").lower():
            continue
        selected.append(_compact_span(s, args.include_structure))
        if len(selected) >= args.max_spans:
            break

    pages_all = sorted({s.get("page_no") for s in spans if s.get("page_no") is not None})
    result = {
        "doc_name":           doc.get("doc_name", args.doc),
        "json_path":          str(json_path),
        "total_spans":        len(spans),
        "total_pages":        len(pages_all),
        "page_range":         [pages_all[0], pages_all[-1]] if pages_all else None,
        "applied_page_filter": sorted(page_filter) if page_filter else None,
        "applied_text_filter": args.filter or None,
        "returned":           len(selected),
        "truncated":          len(selected) >= args.max_spans,
        "spans":              selected,
    }

    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"doc={result['doc_name']}  total_spans={result['total_spans']}  pages={result['total_pages']}")
        print(f"returned={result['returned']}  truncated={result['truncated']}")
        for s in selected:
            print(f"  p{s.get('page_no')}  sz={s.get('size')}  b={s.get('bold')}  {s.get('label')}: {s.get('text', '')[:90]}")


if __name__ == "__main__":
    main()
