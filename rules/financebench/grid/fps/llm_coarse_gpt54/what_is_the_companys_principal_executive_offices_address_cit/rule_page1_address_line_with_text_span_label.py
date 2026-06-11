def rule_page1_address_line_with_text_span_label(doc: dict) -> list[dict]:
    """Match page-1 spans where text is address-like and text_span is exactly the principal executive offices label."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            text_span = (span.get("text_span") or "").strip()
            if span.get("page_no") == 1 and re.search(r'^\(?address of principal executive offices\)?$', text_span, re.I):
                if re.search(r'\d{1,6}\s+\S+.*\b[A-Z][a-z]+', text):
                    out.append(span)
        return out
    except Exception:
        return []
