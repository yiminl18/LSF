def rule_page1_text_with_ein_number_pattern(doc: dict) -> list[dict]:
    """Match page-1 spans containing an EIN-like number pattern near IRS wording."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"\b\d{2}-\d{7}\b", text) and re.search(r"I\.?R\.?S\.?|Employer Identification", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
