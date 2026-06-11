def rule_page1_cover_address_after_company_h1_until_phone(doc: dict) -> list[dict]:
    """Match spans in the page-1 company block from company H1 until the phone number appears."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        started = False
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if span.get("label") == "section_header" and txt and txt.upper() == txt and "FORM 10-" not in txt and "SECURITIES AND EXCHANGE" not in txt:
                started = True
            if started:
                out.append(span)
                if re.search(r'telephone number|^\(\d{3}\)\s*\d{3}', txt, re.I):
                    break
        return out
    except Exception:
        return []
