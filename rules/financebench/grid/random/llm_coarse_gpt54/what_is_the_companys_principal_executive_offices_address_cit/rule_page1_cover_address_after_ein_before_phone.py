def rule_page1_cover_address_after_ein_before_phone(doc: dict) -> list[dict]:
    """Match spans in the page-1 cover block after EIN and before phone number."""
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
                if re.search(r'telephone number', txt, re.I) or re.fullmatch(r'[\(\+\d][\d\(\)\-\+\s]{6,}', txt):
                    break
                out.append(span)
        return out
    except Exception:
        return []
