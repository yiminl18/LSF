import re


def rule_segment_reconciliation_total_assets(doc: dict) -> list[dict]:
    """Reportable-segments reconciliation table that ends with the consolidated Total-assets row (used as a backup when the audited balance sheet has ambiguous column headers)."""
    seg_keywords = (
        "reportable segments",
        "segment information",
        "segments of business",
        "business segment",
    )
    total_assets_row = re.compile(r"\|\s*total\s+assets\s*\|", re.IGNORECASE)
    out = []
    for span in doc.get("texts", []) or []:
        if span.get("label") != "table":
            continue
        text = span.get("text") or ""
        if not total_assets_row.search(text):
            continue
        path = ((span.get("structure") or {}).get("path_text") or "").lower()
        if any(k in path for k in seg_keywords):
            out.append(span)
    return out
