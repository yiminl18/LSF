def rule_page1_phone_with_address_of_principal_executive_offices(doc: dict) -> list[dict]:
    """Match page-1 spans where a phone number appears in the same block as address of principal executive offices."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            blob = (span.get("text", "") or "") + " " + (span.get("text_span", "") or "")
            if span.get("page_no") == 1 and re.search(r"address of principal executive offices", blob, re.I) and re.search(r"(?:\+?\d{1,3}\s*)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}", blob):
                out.append(span)
        return out
    except Exception:
        return []
