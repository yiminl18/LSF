def rule_tables_with_federal_debt_summary_historical(doc: dict) -> list[dict]:
    """Match historical federal debt summary tables such as 'Summary of Federal Debt, Fiscal Years ...'."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") not in {"table", "section_header", "text"}:
                continue
            txt = span.get("text", "") or ""
            if re.search(r'summary of federal debt.*fiscal years', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
