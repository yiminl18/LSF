def rule_page1_cover_page_body_depth2_or_3_outstanding(doc: dict) -> list[dict]:
    """Match page-1 body-depth 2/3 spans with outstanding-share language."""
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            depth = ((span.get("structure") or {}).get("depth") or 0)
            t = (span.get("text") or "").lower()
            if depth in {2, 3} and "outstanding" in t and ("common stock" in t or "shares" in t):
                out.append(span)
    except Exception:
        return []
    return out
