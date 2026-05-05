def rule_page1_san_jose_address(doc: dict) -> list[dict]:
    """Match page-1 spans containing the San Jose address pattern."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = (span.get("text") or "").strip()
            if re.search(r'345 Park Avenue, San Jose, California 95110-2704', text, re.I) or re.search(r'2025 Hamilton Avenue, San Jose, California,?\s*95125', text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
