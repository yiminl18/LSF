def rule_page1_before_irs_employer_identification_no(doc: dict) -> list[dict]:
    """Match address-like spans on page 1 that occur immediately before IRS Employer Identification No. spans."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts[:-1]):
            if span.get("page_no") != 1:
                continue
            t = span.get("text") or ""
            nxt = texts[i+1]
            nxt_full = (nxt.get("text") or "") + " " + (nxt.get("text_span") or "")
            if re.search(r'^\d{1,6}\s+\S+|\bone\b', t, re.I) and re.search(r'street|avenue|drive|plaza|road|market|water|lake|hamilton|rockledge|tower|penn', t, re.I):
                if re.search(r'i\.?r\.?s\.? employer identification|employer identification no', nxt_full, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
