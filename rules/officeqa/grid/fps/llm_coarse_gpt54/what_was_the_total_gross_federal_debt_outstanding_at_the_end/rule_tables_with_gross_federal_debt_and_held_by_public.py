def rule_tables_with_gross_federal_debt_and_held_by_public(doc: dict) -> list[dict]:
    """Match tables containing both gross federal debt wording and held-by-public wording."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r'gross federal debt', text, re.I) and re.search(r'held by the public', text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
