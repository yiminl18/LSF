def rule_page1_contains_month_day_year_and_reporting_keyword(doc: dict) -> list[dict]:
    """Match page-1 spans containing both a reporting keyword and a month-day-year date."""
    import re
    try:
        out = []
        month_re = r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s+\d{4}'
        for span in doc.get("texts", []):
            text = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and re.search(month_re, text):
                if any(k in text for k in [
                    "fiscal year ended",
                    "quarterly period ended",
                    "date of report",
                    "event reported"
                ]):
                    out.append(span)
        return out
    except Exception:
        return []
