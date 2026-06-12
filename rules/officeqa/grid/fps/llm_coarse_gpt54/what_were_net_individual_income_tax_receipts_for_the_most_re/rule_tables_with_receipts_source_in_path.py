def rule_tables_with_receipts_source_in_path(doc: dict) -> list[dict]:
    """Match tables whose structural path already names Budget Receipts by Source."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            if span.get("label") == "table" and re.search(r'Budget Receipts by Source', path, re.I):
                out.append(span)
    except Exception:
        return []
    return out
