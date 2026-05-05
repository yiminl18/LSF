def rule_page1_ein_number_pattern(doc: dict) -> list[dict]:
    """Match page-1 spans containing an EIN-like number pattern."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.search(r"\b\d{2}-\d{7}\b", text):
                out.append(span)
        return out
    except Exception:
        return []
