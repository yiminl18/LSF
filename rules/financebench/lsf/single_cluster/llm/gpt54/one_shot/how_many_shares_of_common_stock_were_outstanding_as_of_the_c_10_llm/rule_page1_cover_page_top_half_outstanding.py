def rule_page1_cover_page_top_half_outstanding(doc: dict) -> list[dict]:
    """Match early page-1 spans with outstanding-share language, reflecting cover-page top-half placement."""
    out = []
    try:
        texts = [s for s in doc.get("texts", []) if s.get("page_no") == 1]
        for idx, span in enumerate(texts[:80]):
            t = (span.get("text") or "").lower()
            if "outstanding" in t and ("common stock" in t or "shares" in t):
                out.append(span)
    except Exception:
        return []
    return out
