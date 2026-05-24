def rule_tables_with_three_year_comparison(doc: dict) -> list[dict]:
    """Match tables that compare three fiscal years and include revenue/sales rows."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            years = set(re.findall(r"\b(20\d{2}|19\d{2})\b", txt))
            if len(years) >= 3 and any(k in txt.lower() for k in ["revenue", "revenues", "sales"]):
                out.append(span)
        return out
    except Exception:
        return []
