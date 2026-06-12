def rule_tables_with_summary_of_federal_debt_and_years(doc: dict) -> list[dict]:
    """Match summary-of-federal-debt tables that also contain multiple year rows."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r'summary of federal debt', text, re.I):
                years = re.findall(r'\b(19[6-9]\d|20[0-2]\d)\b', text)
                if len(set(years)) >= 3:
                    out.append(span)
    except Exception:
        return []
    return out
