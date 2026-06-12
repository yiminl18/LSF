def rule_tables_with_total_gross_federal_debt_outstanding(doc: dict) -> list[dict]:
    """Match tables explicitly mentioning 'total gross federal debt outstanding'."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r'total gross federal debt outstanding', text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
