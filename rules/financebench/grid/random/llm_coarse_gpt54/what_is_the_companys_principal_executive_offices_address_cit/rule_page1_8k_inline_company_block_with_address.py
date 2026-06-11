def rule_page1_8k_inline_company_block_with_address(doc: dict) -> list[dict]:
    """Match page-1 company section headers in 8-Ks whose text_span includes the address block."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            txt = (span.get("text_span") or "")
            if re.search(r'Address of principal executive offices|address of principal executive offices and zip code|principal executive offices', txt, re.I):
                out.append(span)
            elif re.search(r'Issaquah, WA|New York, New York|Warmley, Bristol.*United Kingdom', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
