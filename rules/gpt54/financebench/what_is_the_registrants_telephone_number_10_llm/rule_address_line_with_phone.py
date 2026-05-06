def rule_address_line_with_phone(doc: dict) -> list[dict]:
    """Match spans where the address line and phone number are combined."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r"(address|principal executive offices)", text, re.I) and re.search(r"(\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\+\d{1,3}\s*\d)", text):
                out.append(span)
        return out
    except Exception:
        return []
