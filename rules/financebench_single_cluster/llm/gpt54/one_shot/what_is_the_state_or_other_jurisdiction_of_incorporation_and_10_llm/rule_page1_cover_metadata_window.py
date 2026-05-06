def rule_page1_cover_metadata_window(doc: dict) -> list[dict]:
    """Match a window of spans around the first occurrence of the exact-name-of-registrant label."""
    try:
        texts = doc.get("texts", [])
        idx0 = None
        for i, span in enumerate(texts):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if span.get("page_no") == 1 and "exact name of registrant" in txt:
                idx0 = i
                break
        if idx0 is None:
            return []
        start = max(0, idx0 - 3)
        end = min(len(texts), idx0 + 12)
        return [s for s in texts[start:end] if s.get("page_no") == 1]
    except Exception:
        return []
