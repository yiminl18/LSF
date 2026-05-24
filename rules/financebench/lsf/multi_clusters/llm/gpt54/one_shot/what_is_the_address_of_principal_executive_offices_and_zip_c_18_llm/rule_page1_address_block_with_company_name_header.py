def rule_page1_address_block_with_company_name_header(doc: dict) -> list[dict]:
    """Match company-name header spans on page 1 whose text_span includes the address block."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            head = span.get("text") or ""
            tail = span.get("text_span") or ""
            if re.search(r'(inc\.|corporation|company|plc)', head, re.I) and re.search(r'\d{1,5}\s', tail):
                out.append(span)
        return out
    except Exception:
        return []
