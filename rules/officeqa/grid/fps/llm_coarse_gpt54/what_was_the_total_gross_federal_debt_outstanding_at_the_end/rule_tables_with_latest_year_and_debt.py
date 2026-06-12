def rule_tables_with_latest_year_and_debt(doc: dict) -> list[dict]:
    """Match debt tables containing the latest year in the document's date range."""
    import re
    out = []
    try:
        years = []
        for span in doc.get("texts", []):
            txt = span.get("text", "") or ""
            years += [int(y) for y in re.findall(r'\b(19\d{2}|20\d{2})\b', txt)]
        if not years:
            return []
        max_year = max(years)
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = span.get("text", "") or ""
            if str(max_year) in txt and re.search(r'debt', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
