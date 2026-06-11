def rule_page1_cover_page_all_ein_and_state_candidates(doc: dict) -> list[dict]:
    """Match all page-1 spans that look like either a jurisdiction value or EIN value on the cover page."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text", "") or "").strip()
            if span.get("page_no") != 1:
                continue
            if re.fullmatch(r"\d{2}-\d{7}", txt):
                out.append(span)
            elif re.fullmatch(r"Delaware|Washington|New York|Jersey|Jersey \(Channel Islands\)", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
