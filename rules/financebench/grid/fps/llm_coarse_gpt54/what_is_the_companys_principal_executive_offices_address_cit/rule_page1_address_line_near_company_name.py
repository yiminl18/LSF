def rule_page1_address_line_near_company_name(doc: dict) -> list[dict]:
    """Match address-like spans within a few spans of a large company-name header on page 1."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        company_idxs = []
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("label") in {"section_header", "text"} and (span.get("size") or 0) >= 14:
                company_idxs.append(i)
        for i in company_idxs:
            for cand in texts[max(0, i-2):i+12]:
                if cand.get("page_no") != 1:
                    continue
                ctext = (cand.get("text") or "").strip()
                if re.search(r'\d{1,6}\s+\S+', ctext) and (
                    re.search(r'\b[A-Z][a-z]+,\s*[A-Z]{2}\b', ctext) or
                    re.search(r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+', ctext) or
                    re.search(r'\bUnited Kingdom\b', ctext)
                ):
                    out.append(cand)
        return out
    except Exception:
        return []
