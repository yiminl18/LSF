def rule_tables_with_public_debt_securities_column(doc: dict) -> list[dict]:
    """Match tables containing a 'Public debt securities' column, a strong signal for the answer-bearing table."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r'public debt securities', text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
