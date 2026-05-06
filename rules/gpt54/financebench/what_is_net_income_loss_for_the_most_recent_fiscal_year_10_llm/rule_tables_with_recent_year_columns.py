def rule_tables_with_recent_year_columns(doc: dict) -> list[dict]:
    """Match financial tables that contain recent year columns and a net income row."""
    import re
    try:
        out = []
        years = set()
        for span in doc.get("texts", []):
            txt = span.get("text", "") or ""
            m = re.findall(r"\b(20\d{2})\b", txt)
            years.update(int(x) for x in m)
        recent = sorted(years)[-3:] if years else []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = span.get("text", "") or ""
            if recent and sum(1 for y in recent if str(y) in txt) < 1:
                continue
            if not re.search(r"\bnet income\b|\bnet earnings\b|\bnet loss\b", txt, re.I):
                continue
            out.append(span)
        return out
    except Exception:
        return []
