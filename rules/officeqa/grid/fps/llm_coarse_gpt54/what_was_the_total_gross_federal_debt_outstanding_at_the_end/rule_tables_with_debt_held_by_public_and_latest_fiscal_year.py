def rule_tables_with_debt_held_by_public_and_latest_fiscal_year(doc: dict) -> list[dict]:
    """Match tables containing debt-held-by-public wording and a latest fiscal year row."""
    import re
    out = []
    try:
        years = []
        for span in doc.get("texts", []):
            years += [int(y) for y in re.findall(r'\b(19\d{2}|20\d{2})\b', span.get("text", "") or "")]
        if not years:
            return []
        max_year = max(years)
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = span.get("text", "") or ""
            if str(max_year) in txt and re.search(r'debt held by the public|held by the public', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
