def rule_8k_parenthetical_event_date(doc: dict) -> list[dict]:
    """Match 8-K spans where the answer appears as the parenthetical earliest event date."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'Date of Report.*\([A-Z][a-z]+ \d{1,2}, \d{4}\)', txt):
                out.append(span)
        return out
    except Exception:
        return []
