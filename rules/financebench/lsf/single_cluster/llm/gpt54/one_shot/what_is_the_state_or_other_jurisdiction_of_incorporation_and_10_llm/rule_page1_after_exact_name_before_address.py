def rule_page1_after_exact_name_before_address(doc: dict) -> list[dict]:
    """Match all page-1 spans between the exact-name label and the address label."""
    try:
        texts = doc.get("texts", [])
        start = None
        end = None
        for i, span in enumerate(texts):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if start is None and span.get("page_no") == 1 and "exact name of registrant" in txt:
                start = i
            if start is not None and end is None and span.get("page_no") == 1 and "address of principal executive offices" in txt:
                end = i
                break
        if start is None:
            return []
        if end is None:
            end = min(len(texts), start + 20)
        return [s for s in texts[start:end] if s.get("page_no") == 1]
    except Exception:
        return []
