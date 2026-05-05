def rule_page1_parenthetical_event_date(doc: dict) -> list[dict]:
    """Match 8-K spans with parenthetical earliest-event date on page 1."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if span.get("page_no") == 1 and re.search(r'Date of Report \(Date of earliest event reported\):', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
