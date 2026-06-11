def rule_page1_h2_or_h3_address_headers(doc: dict) -> list[dict]:
    """Match page-1 H2/H3 headers that are likely address fragments in split layouts."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            level = ((span.get("structure") or {}).get("level") or "")
            txt = (span.get("text") or "").strip()
            if level in {"H2", "H3"} and (
                re.search(r'\b\d{5}(?:-\d{4})?\b', txt) or
                re.search(r'^[A-Z]{2,3}$', txt) or
                re.search(r'^\d+\s+\S+', txt) or
                re.search(r'Bristol|Warmley|Seattle|Chicago|Issaquah|San Jose|Santa Monica|St\. Paul|New York', txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
