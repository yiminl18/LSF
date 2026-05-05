def rule_tables_with_two_year_columns_and_total_assets(doc: dict) -> list[dict]:
    """Match tables with total assets and at least two likely year/date columns."""
    import re
    try:
        out = []
        year_re = re.compile(r"\b(20\d{2}|19\d{2})\b")
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text") or ""
            low = text.lower()
            if "total assets" not in low:
                continue
            years = set(year_re.findall(text))
            if len(years) >= 2:
                out.append(span)
        return out
    except Exception:
        return []
