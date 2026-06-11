def rule_page1_phone_after_zip_or_address(doc: dict) -> list[dict]:
    """Match page-1 spans where address/zip and phone number co-occur in the same cover block."""
    try:
        import re
        out = []
        phone_pat = r"(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]?\d{3}[-\s]?\d{4}|\d{10,12})"
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r"(address of principal executive offices|zip code).{0,200}" + phone_pat, txt, re.I | re.S):
                out.append(span)
        return out
    except Exception:
        return []
