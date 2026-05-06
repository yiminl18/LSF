def rule_page1_issaquah_address(doc: dict) -> list[dict]:
    """Match page-1 spans containing the Issaquah Costco address pattern."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r'999 Lake Drive.*Issaquah, WA 98027', text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
