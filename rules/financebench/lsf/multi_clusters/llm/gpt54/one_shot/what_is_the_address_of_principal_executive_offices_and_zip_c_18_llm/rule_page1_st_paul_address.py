def rule_page1_st_paul_address(doc: dict) -> list[dict]:
    """Match page-1 spans containing the 3M St. Paul address pattern."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r'3M Center, St\. Paul, Minnesota', text, re.I) or re.search(r'55144-1000', text):
                out.append(span)
        return out
    except Exception:
        return []
