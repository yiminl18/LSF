def rule_page1_address_and_phone_combined(doc: dict) -> list[dict]:
    """Match page-1 spans combining address and telephone information."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if span.get("page_no") == 1:
                if ("address" in text and "telephone" in text) or re.search(r"address.*area code|telephone number.*principal executive offices", text):
                    out.append(span)
        return out
    except Exception:
        return []
