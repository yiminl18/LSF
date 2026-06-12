def rule_tables_with_gross_federal_debt_outstanding_historical(doc: dict) -> list[dict]:
    """Match historical debt tables that explicitly use the phrase 'gross federal debt outstanding'."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") not in {"table", "text"}:
                continue
            txt = span.get("text", "") or ""
            if re.search(r'gross federal debt outstanding', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
