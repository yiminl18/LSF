def rule_page1_h1_company_with_inline_address(doc: dict) -> list[dict]:
    """Match company H1 section headers on page 1 whose text_span includes the address inline."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            txt = (span.get("text_span") or "")
            if re.search(r'address of principal executive offices|principal executive offices', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
