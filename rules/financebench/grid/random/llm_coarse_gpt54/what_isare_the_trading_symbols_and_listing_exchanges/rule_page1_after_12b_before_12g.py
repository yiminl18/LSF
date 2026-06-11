def rule_page1_after_12b_before_12g(doc: dict) -> list[dict]:
    """Return page-1 spans between the Section 12(b) intro and the Section 12(g) intro."""
    try:
        import re
        texts = doc.get("texts", [])
        start = end = None
        for i, span in enumerate(texts):
            txt = span.get("text", "") or ""
            if start is None and span.get("page_no") == 1 and re.search(r"section 12\(b\) of the act", txt, re.I):
                start = i
            if span.get("page_no") == 1 and re.search(r"section 12\(g\) of the act", txt, re.I):
                end = i
                break
        if start is None:
            return []
        if end is None:
            end = min(len(texts), start + 15)
        return [texts[i] for i in range(start, end) if texts[i].get("page_no") == 1]
    except Exception:
        return []
