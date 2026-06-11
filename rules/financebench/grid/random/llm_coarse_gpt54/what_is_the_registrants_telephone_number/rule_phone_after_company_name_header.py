def rule_phone_after_company_name_header(doc: dict) -> list[dict]:
    """Match phone-number spans shortly after the main company-name header on page 1."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{3}[-/]\d{3}[-/]\d{4})")
        company_idx = None
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                lvl = (span.get("structure", {}) or {}).get("level")
                txt = span.get("text", "") or ""
                if lvl == "H1" and not re.search(r"FORM\s+10-|FORM\s+8-|FORM\s+10-Q|FORM\s+10-K", txt, re.I):
                    company_idx = i
                    break
        if company_idx is None:
            return []
        for j in range(company_idx, min(len(texts), company_idx + 20)):
            cand = texts[j]
            if cand.get("page_no") == 1 and phone_re.search(cand.get("text", "") or ""):
                out.append(cand)
        return out
    except Exception:
        return []
