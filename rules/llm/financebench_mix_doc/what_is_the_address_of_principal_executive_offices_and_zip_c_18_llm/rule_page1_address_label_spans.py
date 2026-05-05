def rule_page1_address_label_spans(doc: dict) -> list[dict]:
    """Match page-1 spans containing the principal executive office address label text."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(r'address of principal executive offices', text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
