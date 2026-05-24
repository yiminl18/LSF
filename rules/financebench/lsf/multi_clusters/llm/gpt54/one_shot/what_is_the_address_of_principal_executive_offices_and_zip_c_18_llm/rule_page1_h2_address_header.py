def rule_page1_h2_address_header(doc: dict) -> list[dict]:
    """Match H2 page-1 section headers that are just the address or postal code block."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            struct = span.get("structure") or {}
            if span.get("page_no") == 1 and span.get("label") == "section_header" and struct.get("level") == "H2":
                text = (span.get("text") or "").strip()
                if re.search(r'^\d{1,5}\s', text) or re.search(r'\b\d{5}(?:-\d{4})?\b', text) or re.search(r'BS30 8XP', text, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
