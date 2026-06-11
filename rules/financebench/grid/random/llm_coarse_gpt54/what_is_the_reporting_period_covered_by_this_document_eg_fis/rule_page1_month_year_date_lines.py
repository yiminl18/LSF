def rule_page1_month_year_date_lines(doc: dict) -> list[dict]:
    """Match page-1 spans with a month-day-year date and nearby report-period wording."""
    import re
    try:
        month = r'(January|February|March|April|May|June|July|August|September|October|November|December)'
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(month + r'\s+\d{1,2},\s+\d{4}', txt):
                if re.search(r'(fiscal year|quarterly period|Date of Report|reported)', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
