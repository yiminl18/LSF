def rule_page1_contains_quarterly_or_fiscal_and_date(doc: dict) -> list[dict]:
    """Match page-1 spans containing fiscal/quarterly wording plus a date-like month-day-year."""
    import re
    try:
        month = r'(January|February|March|April|May|June|July|August|September|October|November|December)'
        date_pat = rf'{month}\s+\d{{1,2}},\s+\d{{4}}'
        out = []
        for span in doc.get("texts", []):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or ""))
            low = txt.lower()
            if span.get("page_no") == 1 and re.search(date_pat, txt, re.I):
                if "fiscal year" in low or "quarterly period" in low or "date of report" in low:
                    out.append(span)
        return out
    except Exception:
        return []
