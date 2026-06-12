def rule_tables_with_federal_debt_and_millions(doc: dict) -> list[dict]:
    """Match Federal Debt tables that are explicitly in millions of dollars."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if re.search(r'federal debt', path + " " + text, re.I) and re.search(r'millions? of dollars|\(in millions', path + " " + text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
