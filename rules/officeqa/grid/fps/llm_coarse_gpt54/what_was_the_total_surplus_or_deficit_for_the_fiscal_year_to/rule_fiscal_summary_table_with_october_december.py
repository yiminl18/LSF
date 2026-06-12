def rule_fiscal_summary_table_with_october_december(doc: dict) -> list[dict]:
    """Match quarter-summary tables with an October-December column and total surplus/deficit row."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'october[-\s]?december', txt, re.I) and re.search(r'total surplus.*deficit', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
