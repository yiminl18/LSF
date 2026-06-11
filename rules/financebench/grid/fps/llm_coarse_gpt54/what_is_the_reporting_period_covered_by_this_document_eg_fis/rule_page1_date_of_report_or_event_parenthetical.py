def rule_page1_date_of_report_or_event_parenthetical(doc: dict) -> list[dict]:
    """Match page-1 date-of-report lines with two dates, where the parenthetical often contains the event date."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = span.get("text") or ""
            low = text.lower()
            if span.get("page_no") == 1 and "date of report" in low:
                if len(re.findall(r'\d{4}', text)) >= 2 or ("(" in text and ")" in text):
                    out.append(span)
        return out
    except Exception:
        return []
