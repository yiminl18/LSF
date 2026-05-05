def rule_page1_h2_address_block_with_phone(doc: dict) -> list[dict]:
    """Match page-1 H2 address blocks whose text or text_span includes the phone number."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if span.get("structure", {}).get("level") != "H2":
                continue
            blob = (span.get("text", "") or "") + " " + (span.get("text_span", "") or "")
            if re.search(r"(address of principal executive offices|telephone number|area code)", blob, re.I) and re.search(r"(?:\+?\d{1,3}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}", blob):
                out.append(span)
        return out
    except Exception:
        return []
