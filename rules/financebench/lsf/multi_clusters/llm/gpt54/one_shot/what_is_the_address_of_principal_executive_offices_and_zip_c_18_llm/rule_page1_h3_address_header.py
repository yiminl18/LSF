def rule_page1_h3_address_header(doc: dict) -> list[dict]:
    """Match H3 page-1 section headers that are fragments of the address block."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            struct = span.get("structure") or {}
            if span.get("page_no") == 1 and span.get("label") == "section_header" and struct.get("level") == "H3":
                text = (span.get("text") or "").strip()
                text_span = (span.get("text_span") or "")
                if re.search(r'^\d{5}(?:-\d{4})?$', text) or re.search(r'address of principal executive offices|zip code', text_span, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
