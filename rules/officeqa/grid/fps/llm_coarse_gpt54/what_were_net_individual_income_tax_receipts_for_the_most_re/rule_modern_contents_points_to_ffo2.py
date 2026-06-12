def rule_modern_contents_points_to_ffo2(doc: dict) -> list[dict]:
    """Match contents tables that mention FFO-2 / receipts by source, useful for locating relevant pages in new docs."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if span.get("label") == "table" and re.search(r'Contents', path, re.I):
                if re.search(r'FFO-2', txt, re.I) or re.search(r'Receipts by Source', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
