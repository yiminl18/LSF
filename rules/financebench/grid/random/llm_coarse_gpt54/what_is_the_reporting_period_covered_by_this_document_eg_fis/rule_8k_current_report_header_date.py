def rule_8k_current_report_header_date(doc: dict) -> list[dict]:
    """Match 8-K current-report header blocks that include the event date line."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'CURRENT REPORT', txt, re.I) and re.search(r'Date of Report', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
