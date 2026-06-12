def rule_tables_with_gross_debt_outstanding_phrase_anywhere(doc: dict) -> list[dict]:
    """Match any table with 'debt outstanding' phrasing."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r'debt outstanding', text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
