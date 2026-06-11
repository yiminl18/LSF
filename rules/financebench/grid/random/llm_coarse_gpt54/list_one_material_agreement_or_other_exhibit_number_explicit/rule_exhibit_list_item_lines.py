def rule_exhibit_list_item_lines(doc: dict) -> list[dict]:
    """Match list_item spans that look like exhibit entries."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "list_item":
                continue
            txt = span.get("text") or ""
            if re.search(r"\bExhibit\s+\d+(?:\.\d+)?[A-Za-z]?\b", txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
