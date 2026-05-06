def rule_page1_before_documents_incorporated(doc: dict) -> list[dict]:
    """Match phone-like spans occurring before the 'DOCUMENTS INCORPORATED BY REFERENCE' area."""
    import re
    try:
        texts = doc.get("texts", [])
        docref_idx = None
        for i, s in enumerate(texts):
            t = ((s.get("text") or "") + " " + (s.get("text_span") or "")).lower()
            if "documents incorporated by reference" in t:
                docref_idx = i
                break
        if docref_idx is None:
            docref_idx = len(texts)
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b|\b\d{3}[-]\d{4}[-]\d{4}\b)")
        for span in texts[:docref_idx]:
            if span.get("page_no") == 1:
                text = (span.get("text") or "") + " " + (span.get("text_span") or "")
                if phone_re.search(text):
                    out.append(span)
        return out
    except Exception:
        return []
