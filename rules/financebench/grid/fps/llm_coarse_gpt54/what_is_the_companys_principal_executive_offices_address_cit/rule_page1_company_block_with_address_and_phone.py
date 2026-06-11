def rule_page1_company_block_with_address_and_phone(doc: dict) -> list[dict]:
    """Match page-1 company identity block spans containing both address and phone context."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r'registrant.?s telephone number', txt, re.I) and (
                re.search(r'address of principal executive offices', txt, re.I) or
                re.search(r'\d{1,6}\s+\S+.*\b[A-Z][a-z]+,\s*[A-Z]{2}\s+\d{5}', txt) or
                re.search(r'\d{1,6}\s+\S+.*\b[A-Z][a-z]+,\s*[A-Z][a-z]+', txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
