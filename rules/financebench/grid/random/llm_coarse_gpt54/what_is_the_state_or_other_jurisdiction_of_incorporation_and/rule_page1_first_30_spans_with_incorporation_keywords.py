def rule_page1_first_30_spans_with_incorporation_keywords(doc: dict) -> list[dict]:
    """Match early page-1 spans near the top containing incorporation or IRS keywords."""
    try:
        import re
        out = []
        for span in doc.get("texts", [])[:30]:
            text = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"incorporation|organization|Employer Identification|I\.?R\.?S\.?", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
