def rule_page1_company_block_with_exact_name_and_address(doc: dict) -> list[dict]:
    """Match large company-name section headers on page 1 whose text_span includes the address block."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            txt = span.get("text") or ""
            ts = span.get("text_span") or ""
            if re.search(r'(INC\.|CORPORATION|PLC|COMPANY)', txt, re.I) and re.search(r'\d{2,}.*(?:\d{5}(?:-\d{4})?|BS30 ?8XP)', ts, re.I):
                out.append(span)
        return out
    except Exception:
        return []
