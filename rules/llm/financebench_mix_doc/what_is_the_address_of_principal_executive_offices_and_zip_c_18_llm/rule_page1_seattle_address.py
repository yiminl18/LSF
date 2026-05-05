def rule_page1_seattle_address(doc: dict) -> list[dict]:
    """Match page-1 spans containing the Seattle address pattern."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = (span.get("text") or "").strip()
            if re.search(r'410 Terry Avenue North.*Seattle, Washington 98109-5210', text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
