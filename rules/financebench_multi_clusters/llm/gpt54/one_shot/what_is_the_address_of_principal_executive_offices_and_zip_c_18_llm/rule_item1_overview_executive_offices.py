def rule_item1_overview_executive_offices(doc: dict) -> list[dict]:
    """Match Item 1/Business overview spans mentioning executive offices are located at."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = (((span.get("structure") or {}).get("path_text")) or "")
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'item 1|business|overview', path, re.I) and re.search(r'executive offices.*located at', text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
