def rule_page2_text_span_contains_outstanding(doc: dict) -> list[dict]:
    """Match any page-2 span whose text or text_span contains outstanding-share language."""
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("page_no") != 2:
                continue
            full = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if "outstanding" in full and ("common stock" in full or "shares" in full):
                out.append(span)
    except Exception:
        return []
    return out
