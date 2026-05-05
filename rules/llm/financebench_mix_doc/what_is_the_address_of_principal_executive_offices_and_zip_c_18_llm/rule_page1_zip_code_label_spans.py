def rule_page1_zip_code_label_spans(doc: dict) -> list[dict]:
    """Match page-1 spans containing ZIP code label text near the address."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(r'\bzip code\b', text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
