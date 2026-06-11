def rule_page1_cover_address_after_ein_until_zip(doc: dict) -> list[dict]:
    """Match spans in the page-1 cover block after EIN until zip code marker."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        started = False
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if re.search(r'Employer Identification No|I\.R\.S\.', txt, re.I):
                started = True
                continue
            if started:
                out.append(span)
                if re.search(r'Zip Code', txt, re.I):
                    break
        return out
    except Exception:
        return []
