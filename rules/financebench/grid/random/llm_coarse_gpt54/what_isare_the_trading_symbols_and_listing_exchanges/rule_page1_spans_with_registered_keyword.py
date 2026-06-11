def rule_page1_spans_with_registered_keyword(doc: dict) -> list[dict]:
    """Match page-1 spans containing 'registered' near listing-related terms."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"registered", txt, re.I) and (
                re.search(r"exchange", txt, re.I) or re.search(r"trading symbol", txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
