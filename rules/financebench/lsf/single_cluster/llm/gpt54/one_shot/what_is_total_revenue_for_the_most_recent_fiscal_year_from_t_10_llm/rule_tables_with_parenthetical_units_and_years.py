def rule_tables_with_parenthetical_units_and_years(doc: dict) -> list[dict]:
    """Match financial statement tables that include year columns and unit notes like millions/billions."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            years = re.findall(r"\b(20\d{2}|19\d{2})\b", txt)
            if len(set(years)) >= 2 and any(u in txt for u in ["million", "millions", "billion", "billions"]):
                out.append(span)
        return out
    except Exception:
        return []
