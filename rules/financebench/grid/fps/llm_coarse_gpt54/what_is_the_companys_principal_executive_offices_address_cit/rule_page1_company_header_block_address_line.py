def rule_page1_company_header_block_address_line(doc: dict) -> list[dict]:
    """Match page-1 body/header spans in the registrant identity block that contain a street plus city/state."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if re.search(r'\d{1,6}\s+\S+', txt) and (
                re.search(r',\s*[A-Z][a-z]+,\s*[A-Z]{2}\b', txt) or
                re.search(r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', txt) or
                re.search(r'\b[A-Z][a-z]+,\s*[A-Z]{2}\s+\d{5}', txt) or
                re.search(r'\bUnited Kingdom\b|\bUnited States\b|\bNew Jersey\b|\bCalifornia\b|\bVirginia\b|\bNew York\b', txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
