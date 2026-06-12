def rule_tables_with_year_end_rows(doc: dict) -> list[dict]:
    """Match tables containing annual rows like 1980, 1981, 1982 etc., where the answer is often the latest annual debt value."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            years = re.findall(r'\b(19[7-9]\d|20[0-2]\d)\b', text)
            if len(set(years)) >= 4:
                out.append(span)
    except Exception:
        return []
    return out
