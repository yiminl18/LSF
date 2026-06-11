def rule_page1_h2_header_address_candidate(doc: dict) -> list[dict]:
    """Match page-1 H2 section headers that themselves are address lines."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            if ((span.get("structure") or {}).get("level") or "") != "H2":
                continue
            txt = span.get("text") or ""
            if re.search(r'^\d{2,} ', txt) or re.search(r'\b(?:Bristol|California|Washington|New York|Minnesota|CA|WA)\b', txt):
                out.append(span)
        return out
    except Exception:
        return []
