def rule_page1_address_before_zip_or_ein(doc: dict) -> list[dict]:
    """Match page-1 address-like spans that also mention zip code or IRS/Employer ID in the same combined text."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r'(zip code|i\.?r\.?s\.? employer identification|irs employer identification|employer identification no)', txt, re.I):
                if re.search(r'\b(address of principal executive offices)\b', txt, re.I) or re.search(r'\d{3,}.*?,?\s+[A-Z][a-z]+.*?,?\s+[A-Z]{2,}', txt):
                    out.append(span)
        return out
    except Exception:
        return []
