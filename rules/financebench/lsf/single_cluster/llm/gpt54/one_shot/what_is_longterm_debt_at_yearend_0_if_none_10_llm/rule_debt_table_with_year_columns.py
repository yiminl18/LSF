def rule_debt_table_with_year_columns(doc: dict) -> list[dict]:
    """Match debt tables that have long-term debt rows and multiple year/date columns."""
    import re
    try:
        out = []
        year_re = re.compile(r"\b(20\d{2}|19\d{2})\b")
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            joined = " ".join((c.get("text") or "") for c in cells)
            if re.search(r"\blong[\-\s]?term debt\b", joined.lower()):
                years = set(year_re.findall(joined))
                if len(years) >= 1:
                    out.append(span)
        return out
    except Exception:
        return []
