def rule_page1_amazon_seattle_header(doc: dict) -> list[dict]:
    """Match Amazon-style page-1 address line with Seattle, Washington."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.search(r'410 Terry Avenue North.*Seattle,\s*Washington\s*98109', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
