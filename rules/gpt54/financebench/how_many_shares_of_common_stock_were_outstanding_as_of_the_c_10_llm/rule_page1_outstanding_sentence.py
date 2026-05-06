def rule_page1_outstanding_sentence(doc: dict) -> list[dict]:
    """Match page-1 spans containing both 'outstanding' and 'common stock/shares' language."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            if span.get("page_no") == 1:
                t = text.lower()
                if "outstanding" in t and ("common stock" in t or "common shares" in t or "shares of common stock" in t):
                    out.append(span)
    except Exception:
        return []
    return out
