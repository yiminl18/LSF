def rule_page1_event_report_date_pattern(doc: dict) -> list[dict]:
    """Match page-1 8-K spans with a report date followed by a parenthetical event date."""
    import re
    try:
        month = r'(January|February|March|April|May|June|July|August|September|October|November|December)'
        pat = rf'Date of Report .*?:\s*{month}\s+\d{{1,2}},\s+\d{{4}}\s*\(\s*{month}\s+\d{{1,2}},\s+\d{{4}}\s*\)'
        out = []
        for span in doc.get("texts", []):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or ""))
            if span.get("page_no") == 1 and re.search(pat, txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
