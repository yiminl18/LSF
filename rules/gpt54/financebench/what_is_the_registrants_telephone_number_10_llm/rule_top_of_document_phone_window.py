def rule_top_of_document_phone_window(doc: dict) -> list[dict]:
    """Match phone-like spans in the first 40 spans of the document, where the cover-page answer lives."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s\-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}-\d{3}-\d{4}\b)")
        for span in doc.get("texts", [])[:40]:
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and phone_re.search(text):
                out.append(span)
        return out
    except Exception:
        return []
