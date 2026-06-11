def rule_8k_parenthetical_earliest_event_date(doc: dict) -> list[dict]:
    """Match 8-K date lines that include a parenthetical earliest-event date."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = span.get("text") or ""
            low = text.lower()
            if "date of report" in low and "earliest event reported" in low:
                if "(" in text and ")" in text:
                    out.append(span)
        return out
    except Exception:
        return []
