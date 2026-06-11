def rule_page1_cover_address_with_postal_code_or_zip(doc: dict) -> list[dict]:
    """Match page-1 spans containing a location plus postal code/zip."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if (
                re.search(r'(Issaquah|Seattle|San Jose|New York|St\. Paul|Chicago|Santa Monica).*\b\d{5}(?:-\d{4})?\b', txt, re.I) or
                re.search(r'(Warmley|Bristol).*\bBS\d', txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
