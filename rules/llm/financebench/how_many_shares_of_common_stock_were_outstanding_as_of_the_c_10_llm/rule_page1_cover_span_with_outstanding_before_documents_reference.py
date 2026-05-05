def rule_page1_cover_span_with_outstanding_before_documents_reference(doc: dict) -> list[dict]:
    """Match page-1/2 spans with outstanding-share language before the documents-incorporated section."""
    try:
        texts = doc.get("texts", [])
        out = []
        stop_idx = None
        for i, span in enumerate(texts):
            if "documents incorporated by reference" in (span.get("text") or "").lower():
                stop_idx = i
                break
        if stop_idx is None:
            stop_idx = len(texts)
        for span in texts[:stop_idx]:
            if span.get("page_no") in (1, 2):
                t = (span.get("text") or "").lower()
                if "outstanding" in t and ("common stock" in t or "shares" in t):
                    out.append(span)
        return out
    except Exception:
        return []
