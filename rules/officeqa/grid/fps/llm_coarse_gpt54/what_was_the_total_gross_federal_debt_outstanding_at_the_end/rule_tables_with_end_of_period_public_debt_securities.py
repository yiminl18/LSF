def rule_tables_with_end_of_period_public_debt_securities(doc: dict) -> list[dict]:
    """Match end-of-period balance tables with public debt securities columns."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = span.get("text", "") or ""
            if re.search(r'end of period', txt, re.I) and re.search(r'public debt securities', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
