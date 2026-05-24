def rule_table_with_net_income_and_three_years(doc: dict) -> list[dict]:
    """Match tables with a net income row and at least three distinct year labels."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = span.get("text", "") or ""
            years = set(re.findall(r"\b20\d{2}\b", txt))
            if len(years) < 2:
                continue
            if re.search(r"\bnet income\b|\bnet earnings\b|\bnet loss\b", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
