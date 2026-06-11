def rule_page1_address_like_section_header_with_path_company(doc: dict) -> list[dict]:
    """Match page-1 section headers under the company path that look like address or location fragments."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "").strip()
            if path and path != txt and (
                re.search(r'\b\d+\b', txt) or
                re.search(r'Santa Monica|San Jose|Seattle|Issaquah|Chicago|St\. Paul|New York|Warmley|Bristol', txt, re.I) or
                re.fullmatch(r'[A-Z]{2,3}', txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
