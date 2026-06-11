def rule_page1_any_month_day_year_with_ended_context(doc: dict) -> list[dict]:
    """Match page-1 spans containing a month-day-year date near 'ended' or 'reported'."""
    import re
    try:
        month = r'(January|February|March|April|May|June|July|August|September|October|November|December)'
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(month + r'\s+\d{1,2},\s+\d{4}', txt):
                if re.search(r'(ended|reported|Date of Report|earliest event)', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
