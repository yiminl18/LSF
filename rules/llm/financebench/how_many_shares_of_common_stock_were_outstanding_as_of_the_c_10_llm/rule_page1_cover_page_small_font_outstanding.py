def rule_page1_cover_page_small_font_outstanding(doc: dict) -> list[dict]:
    """Match small-font page-1 spans with outstanding-share language, common on SEC cover pages."""
    out = []
    try:
        for span in doc.get("texts", []):
            size = span.get("size") or 0
            t = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and size <= 9.5 and "outstanding" in t and ("common stock" in t or "shares" in t):
                out.append(span)
    except Exception:
        return []
    return out
