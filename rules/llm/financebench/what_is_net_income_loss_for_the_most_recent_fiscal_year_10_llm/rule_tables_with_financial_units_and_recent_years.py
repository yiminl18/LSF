def rule_tables_with_financial_units_and_recent_years(doc: dict) -> list[dict]:
    """Match financial tables using units like million/billion and containing year columns."""
    import re
    try:
        out = []
        for s in doc.get("texts", []):
            if s.get("label") != "table":
                continue
            txt = s.get("text", "") or ""
            if re.search(r"million|billion|\$", txt, re.I) and len(set(re.findall(r"\b20\d{2}\b", txt))) >= 1:
                if re.search(r"\bnet income\b|\bnet earnings\b|\bnet loss\b", txt, re.I):
                    out.append(s)
        return out
    except Exception:
        return []
