def rule_page1_company_header_followed_by_address_block(doc: dict) -> list[dict]:
    """Match spans in the first company-identification block on page 1 that contain the office address."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        company_idx = None
        for i, span in enumerate(texts):
            if span.get("page_no") == 1:
                t = (span.get("text") or "") + " " + (span.get("text_span") or "")
                if re.search(r'exact name of registrant', t, re.I):
                    company_idx = i
                    break
        if company_idx is None:
            return []
        for span in texts[company_idx: min(len(texts), company_idx + 12)]:
            t = (span.get("text") or "").strip()
            if re.search(r'^\d{1,6}\s+\S+|\bone\b', t, re.I) and re.search(r'street|avenue|drive|plaza|road|market|water|lake|hamilton|rockledge|tower|penn', t, re.I):
                out.append(span)
        return out
    except Exception:
        return []
