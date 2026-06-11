def rule_cover_page_phone_in_company_intro_paragraph(doc: dict) -> list[dict]:
    """Match page-1 company intro spans that include the registrant phone number inline."""
    try:
        import re
        out = []
        phone_pat = re.compile(r"(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]?\d{3}[-\s]?\d{4}|\d{10,12})")
        for span in doc.get("texts", []):
            if span.get("page_no") == 1:
                txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
                if phone_pat.search(txt) and re.search(r"exact name of registrant|state or other jurisdiction|address of principal executive offices|commission file|employer identification", txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
