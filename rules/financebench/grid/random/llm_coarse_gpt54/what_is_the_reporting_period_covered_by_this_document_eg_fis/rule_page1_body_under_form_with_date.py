def rule_page1_body_under_form_with_date(doc: dict) -> list[dict]:
    """Match page-1 body-level spans under FORM paths that contain a month-day-year date."""
    import re
    try:
        month = r'(January|February|March|April|May|June|July|August|September|October|November|December)'
        out = []
        for span in doc.get("texts", []):
            level = ((span.get("structure") or {}).get("level") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and level == "Body" and re.search(r'FORM\s+10-(K|Q|8-K)', path, re.I):
                if re.search(month + r'\s+\d{1,2},\s+\d{4}', txt):
                    out.append(span)
        return out
    except Exception:
        return []
