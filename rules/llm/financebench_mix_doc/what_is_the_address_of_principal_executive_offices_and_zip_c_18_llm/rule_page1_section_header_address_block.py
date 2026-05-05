def rule_page1_section_header_address_block(doc: dict) -> list[dict]:
    """Match page-1 section headers that themselves are the address block."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            text = (span.get("text") or "").strip()
            if re.search(r'^\d{1,5}\s', text) or re.search(r'\b\d{5}(?:-\d{4})?\b', text):
                if re.search(r'United Kingdom|[A-Z]{2}\s+\d{5}|California \d{5}|Washington \d{5}|Minnesota \d{5}|Bristol', text, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
