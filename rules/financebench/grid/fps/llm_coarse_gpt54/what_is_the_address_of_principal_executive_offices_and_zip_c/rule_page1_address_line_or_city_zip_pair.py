def rule_page1_address_line_or_city_zip_pair(doc: dict) -> list[dict]:
    """Match either the street line or the city/ZIP line when the address is split across two consecutive spans."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts)-1):
            a = texts[i]
            b = texts[i+1]
            if a.get("page_no") != 1 or b.get("page_no") != 1:
                continue
            at = (a.get("text") or "").strip()
            bt = (b.get("text") or "").strip()
            if re.search(r'^\d{1,6}\s+\S+|\bone\b', at, re.I) and re.search(r'street|avenue|drive|plaza|road|market|water|lake|hamilton|rockledge|tower|penn', at, re.I):
                out.append(a)
                if re.search(r'\b\d{5}(?:-\d{4})?\b', bt) or re.search(r'\b[A-Z][A-Za-z\.\- ]+,\s*[A-Z]{2,}\b', bt):
                    out.append(b)
        return out
    except Exception:
        return []
