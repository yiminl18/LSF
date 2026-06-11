def rule_page1_address_like_section_header_under_company(doc: dict) -> list[dict]:
    """Match page-1 section headers under the company block whose text is an address line."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "")
            t = (span.get("text") or "").strip()
            if re.search(r'company|corporation|inc\.|plc|incorporated', path, re.I):
                if re.search(r'^\d{1,6}\s+\S+|\bone\b', t, re.I) and re.search(r'street|avenue|drive|plaza|road|market|water|lake|hamilton|rockledge|tower|penn', t, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
