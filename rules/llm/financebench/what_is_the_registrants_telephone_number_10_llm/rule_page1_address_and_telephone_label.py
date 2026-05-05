def rule_page1_address_and_telephone_label(doc: dict) -> list[dict]:
    """Match page-1 spans with the combined address-and-telephone label used by some filers."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(r"address\s+and\s+telephone\s+number.*principal executive offices", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
