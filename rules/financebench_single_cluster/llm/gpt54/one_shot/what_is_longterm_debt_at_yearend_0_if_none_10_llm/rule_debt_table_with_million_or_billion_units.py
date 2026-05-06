def rule_debt_table_with_million_or_billion_units(doc: dict) -> list[dict]:
    """Match debt tables that include long-term debt and unit labels like million or billion."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if re.search(r"\blong[\-\s]?term debt\b", txt) and ("million" in txt or "billion" in txt):
                out.append(span)
        return out
    except Exception:
        return []
