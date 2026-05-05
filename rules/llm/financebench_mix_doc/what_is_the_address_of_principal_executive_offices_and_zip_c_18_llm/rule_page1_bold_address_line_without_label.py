def rule_page1_bold_address_line_without_label(doc: dict) -> list[dict]:
    """Match bold page-1 address-looking spans even when the label is in a neighboring span."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("bold") != 1:
                continue
            text = (span.get("text") or "").strip()
            if re.search(r'^\d{1,5}\s', text) and (
                ',' in text or re.search(r'\b[A-Z]{2}\b', text) or re.search(r'United Kingdom', text, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
