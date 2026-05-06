def rule_page1_security_registration_cluster(doc: dict) -> list[dict]:
    """Match the cluster of page 1 spans between Section 12(b) and Section 12(g)."""
    try:
        texts = doc.get("texts", [])
        start = end = None
        for i, span in enumerate(texts):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if start is None and "section 12(b)" in txt:
                start = i
            if start is not None and "section 12(g)" in txt:
                end = i
                break
        if start is None:
            return []
        if end is None:
            end = min(len(texts), start + 20)
        return [s for s in texts[start:end+1] if s.get("page_no") == 1]
    except Exception:
        return []
