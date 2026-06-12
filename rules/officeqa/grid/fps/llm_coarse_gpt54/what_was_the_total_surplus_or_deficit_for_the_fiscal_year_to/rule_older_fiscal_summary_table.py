def rule_older_fiscal_summary_table(doc: dict) -> list[dict]:
    """Match older Treasury Bulletin fiscal summary tables with 'Fiscal year or month' and 'Total surplus or deficit'."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'fiscal year or month', txt, re.I) and re.search(r'total surplus or deficit', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
