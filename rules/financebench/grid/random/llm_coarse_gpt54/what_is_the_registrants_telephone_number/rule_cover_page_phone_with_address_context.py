def rule_cover_page_phone_with_address_context(doc: dict) -> list[dict]:
    """Match phone-number spans on page 1 that also mention address, zip, or executive-office context."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{3}[-/]\d{3}[-/]\d{4})")
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if span.get("page_no") != 1:
                continue
            if phone_re.search(text) and re.search(r"address|executive offices|zip code|principal", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
