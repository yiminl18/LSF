def rule_tables_with_net_column_and_recent_fiscal_year(doc: dict) -> list[dict]:
    """Match detailed receipt tables with Net columns and recent fiscal year rows, where the answer is usually the latest quarter/month net value."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'\bNet\b', txt, re.I) and re.search(r'Fiscal\s+\d{4}|20\d{2}|19\d{2}', txt):
                if re.search(r'Individual', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
