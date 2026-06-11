def rule_page1_first_address_like_after_company_name(doc: dict) -> list[dict]:
    """Match the first address-like span after a company-name heading on page 1."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        company_positions = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            if span.get("label") in {"section_header", "text"}:
                t = (span.get("text") or "").strip()
                if re.search(r'(inc\.|incorporated|corporation|plc)$', t, re.I) or re.search(r'^[A-Z][A-Za-z&\.\' ]+(Inc\.|Corporation|plc|Incorporated)$', t):
                    company_positions.append(i)
        for idx in company_positions:
            for span in texts[idx+1: min(len(texts), idx+10)]:
                if span.get("page_no") != 1:
                    continue
                t = (span.get("text") or "").strip()
                if re.search(r'^\d{1,6}\s+\S+|\bone\b', t, re.I) and re.search(r'street|avenue|drive|plaza|road|market|water|lake|hamilton|rockledge|tower|penn', t, re.I):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
