def rule_page1_or_2_cover_page_outstanding(doc: dict) -> list[dict]:
    """Match page 1 or 2 spans with cover-page outstanding-share language."""
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("page_no") not in {1, 2}:
                continue
            t = (span.get("text") or "").lower()
            if "outstanding" in t and ("common stock" in t or "shares" in t):
                out.append(span)
    except Exception:
        return []
    return out
