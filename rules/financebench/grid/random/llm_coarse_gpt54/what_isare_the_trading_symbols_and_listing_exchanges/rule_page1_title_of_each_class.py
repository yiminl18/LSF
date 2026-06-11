def rule_page1_title_of_each_class(doc: dict) -> list[dict]:
    """Match page-1 spans containing 'Title of each class' or close variants."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"title of each class", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
