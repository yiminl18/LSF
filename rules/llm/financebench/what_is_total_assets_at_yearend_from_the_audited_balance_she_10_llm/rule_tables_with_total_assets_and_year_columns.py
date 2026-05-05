def rule_tables_with_total_assets_and_year_columns(doc: dict) -> list[dict]:
    """Match tables containing total assets and likely year-end columns."""
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
            years = year_re.findall(text)
            if len(set(years)) >= 1:
                out.append(span)
        return out
    except Exception:
        return []
