def rule_tables_on_pages_named_federal_fiscal_operations(doc: dict) -> list[dict]:
    """Match tables on pages/paths associated with Federal Fiscal Operations."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "")
            if span.get("label") == "table":
                if re.search(r'Federal Fiscal Operations', path, re.I) or re.search(r'Federal Fiscal Operations', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
