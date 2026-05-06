def rule_tables_with_two_or_more_year_columns(doc: dict) -> list[dict]:
    """Match financial statement tables with at least two year/date columns."""
    import re
    try:
        out = []
        year_re = re.compile(r"\b(20\d{2}|19\d{2})\b")
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            years = set()
            for c in cells:
                for y in year_re.findall(c.get("text") or ""):
                    years.add(y)
            if len(years) >= 2 and "assets" in (span.get("text") or "").lower():
                out.append(span)
        return out
    except Exception:
        return []
