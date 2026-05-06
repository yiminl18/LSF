def rule_cover_page_phone_with_address(doc: dict) -> list[dict]:
    """Match cover-page spans containing both an address-like phrase and a phone number."""
    import re
    try:
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b|\b\d{3}[-]\d{4}[-]\d{4}\b)")
        for span in doc.get("texts", []):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or ""))
            low = text.lower()
            if span.get("page_no") == 1 and phone_re.search(text):
                if "address of principal executive offices" in low or "principal executive offices" in low or "address and telephone number" in low:
                    out.append(span)
        return out
    except Exception:
        return []
