def rule_page1_current_report_event_date(doc: dict) -> list[dict]:
    """Match page-1 spans in current reports that include the event date."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(r'CURRENT REPORT', txt, re.I):
                if re.search(r'Date of Report.*event reported', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
