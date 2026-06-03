#!/usr/bin/env python3
"""Convert officeqa raw layout JSON (`document.elements`) into the reconstructed
`texts`-span schema the LSF pipeline expects (court/nopv format).

The pipeline (rule_gen prompt, generated rule functions, rule_apply.merge,
precompute) all read `doc["texts"]` where each span has at least
`text, label, page_no, structure{level, level_index, path_text, ...}` plus the
typographic flags (`bold, size, ...`). officeqa's raw format has none of that
shape, so this converter maps it over.

Lossless: every officeqa element field that has no texts-schema home is RETAINED
on the span (`officeqa_type`, `officeqa_id`, `bbox`, `description`), and the
top-level `document.pages` / `metadata` / `error_status` are kept under
`officeqa_*` keys. Nothing from the source is dropped.

Note on missing signal: officeqa has no font/size/bold/structure, so those are
filled with defaults (0 / "" / synthesized). `structure.path_text` is
reconstructed by tracking the most recent heading element.

Usage:
  python scripts/convert_officeqa_to_texts.py IN.json OUT.json        # one file
  python scripts/convert_officeqa_to_texts.py --in-dir D --out-dir D  # batch
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# officeqa element `type` -> texts `label` (nearest court/nopv equivalent).
# The original type is always retained as `officeqa_type`, so this is lossless.
LABEL_MAP = {
    "text": "text",
    "section_header": "section_header",
    "title": "section_header",
    "table": "table",
    "caption": "caption",
    "footnote": "text",
    "list_item": "list_item",
    # page_number / page_header / page_footer / figure: keep the original type as
    # the label (rules simply won't match these, which is the correct behavior).
}
HEADING_TYPES = {"title", "section_header"}


def convert_doc(raw: dict, doc_name: str) -> dict:
    document = raw.get("document", {}) or {}
    elements = document.get("elements", []) or []

    texts: list[dict] = []
    cur_section = ""     # running breadcrumb: most recent heading content
    level_index = 0

    for el in elements:
        content = el.get("content")
        etype = el.get("type", "text") or "text"
        bbox = el.get("bbox") or []
        page_id = bbox[0].get("page_id") if bbox else None
        label = LABEL_MAP.get(etype, etype)
        is_heading = etype in HEADING_TYPES

        if is_heading and content:
            cur_section = content.strip()
            level_index += 1

        span = {
            # ── texts-schema fields the pipeline reads ──────────────────────
            "text": content or "",
            "text_span": "",
            "label": label,
            "page_no": page_id if page_id is not None else 0,
            "size": 0.0,
            "bold": 0,
            "font": "",
            "all_cap": 0,
            "num_st": 0,
            "is_center": 0,
            "is_underline": 0,
            "structure": {
                "level": "H1" if is_heading else "Body",
                "level_index": level_index,
                "parent_id": None,
                "path_text": cur_section,
                "depth": 1,
            },
            # ── retained officeqa originals (lossless) ──────────────────────
            "officeqa_type": etype,
            "officeqa_id": el.get("id"),
            "bbox": bbox,
            "description": el.get("description"),
        }
        texts.append(span)

    return {
        "doc_name": doc_name,
        "source_format": "officeqa",
        "texts": texts,
        # retained top-level officeqa info
        "officeqa_pages": document.get("pages"),
        "officeqa_error_status": raw.get("error_status"),
        "officeqa_metadata": raw.get("metadata"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", nargs="?", help="input officeqa json")
    ap.add_argument("out", nargs="?", help="output texts-schema json")
    ap.add_argument("--in-dir", help="batch: input dir of officeqa jsons")
    ap.add_argument("--out-dir", help="batch: output dir")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    if args.in_dir:
        ind, outd = Path(args.in_dir), Path(args.out_dir)
        outd.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(ind.glob("*.json")):
            conv = convert_doc(json.loads(f.read_text(encoding="utf-8")), f.stem)
            (outd / f.name).write_text(json.dumps(conv), encoding="utf-8")
            n += 1
        print(f"converted {n} files -> {outd}")
    else:
        conv = convert_doc(json.loads(Path(args.inp).read_text(encoding="utf-8")),
                           Path(args.inp).stem)
        txt = json.dumps(conv, indent=2) if args.pretty else json.dumps(conv)
        Path(args.out).write_text(txt, encoding="utf-8")
        print(f"converted {args.inp} -> {args.out}  ({len(conv['texts'])} spans)")


if __name__ == "__main__":
    main()
