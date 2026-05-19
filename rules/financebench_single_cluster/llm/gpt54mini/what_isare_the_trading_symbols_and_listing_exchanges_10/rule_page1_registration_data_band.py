def rule_page1_registration_data_band(doc: dict) -> list[dict]:
    """Match the dense band of page 1 body/header spans between company identity and 'Documents Incorporated by Reference'."""
    try:
        texts = doc.get("texts", [])
        start = None
        end = None
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if start is None and ("exact name of registrant" in txt or "exact name of registrant as specified in its charter" in txt):
                start = max(0, i - 1)
            if start is not None and "documents incorporated by reference" in txt:
                end = i
                break
        if start is None:
            return []
        if end is None:
            end = min(len(texts) - 1, start + 40)
        return [s for s in texts[start:end+1] if s.get("page_no") == 1]
    except Exception:
        return []
