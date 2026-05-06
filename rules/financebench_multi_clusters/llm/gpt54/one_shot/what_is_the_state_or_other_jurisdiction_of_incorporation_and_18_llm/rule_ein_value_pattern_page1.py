def rule_ein_value_pattern_page1(doc: dict) -> list[dict]:
    """Match page-1 spans containing an EIN-like number pattern."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"\b\d{2}-\d{7}\b", text):
                out.append(span)
        return out
    except Exception:
        return []
