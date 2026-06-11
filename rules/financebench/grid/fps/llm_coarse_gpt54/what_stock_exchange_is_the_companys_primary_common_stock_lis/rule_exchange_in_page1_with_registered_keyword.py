def rule_exchange_in_page1_with_registered_keyword(doc: dict) -> list[dict]:
    """Match page-1 spans containing both 'registered' and an exchange name."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            combined = " ".join([span.get("text", "") or "", span.get("text_span", "") or ""])
            if span.get("page_no") == 1 and re.search(r'registered', combined, re.I) and re.search(r'new york stock exchange|nasdaq|global select market', combined, re.I):
                out.append(span)
        return out
    except Exception:
        return []
