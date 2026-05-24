def rule_phone_with_address_and_zip(doc: dict) -> list[dict]:
    """Match spans that contain both address/zip context and a phone number on the cover page."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s\-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}-\d{3}-\d{4}\b)")
        for span in doc.get("texts", []):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and phone_re.search(text):
                if re.search(r"address|zip code|principal executive offices|new brunswick|seattle|bethesda|beaverton|corning|san jose|issaquah|bristol", text, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
