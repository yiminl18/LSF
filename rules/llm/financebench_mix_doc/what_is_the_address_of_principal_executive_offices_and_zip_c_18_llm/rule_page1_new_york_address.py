def rule_page1_new_york_address(doc: dict) -> list[dict]:
    """Match page-1 spans containing the New York address pattern."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = (span.get("text") or "").strip()
            if re.search(r'330 West 34th Street, New York, New York 10001', text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
