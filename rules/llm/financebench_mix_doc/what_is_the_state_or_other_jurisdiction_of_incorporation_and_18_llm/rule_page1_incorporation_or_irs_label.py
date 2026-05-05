def rule_page1_incorporation_or_irs_label(doc: dict) -> list[dict]:
    """Match page-1 spans containing the incorporation or IRS identification labels."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"state or other jurisdiction of incorporation|i\.?r\.?s\.? employer identification", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
