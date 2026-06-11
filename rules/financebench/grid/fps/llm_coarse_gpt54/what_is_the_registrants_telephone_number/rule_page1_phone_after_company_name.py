def rule_page1_phone_after_company_name(doc: dict) -> list[dict]:
    """Match phone-like spans appearing after the main company name on page 1."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        phone_pat = re.compile(r"(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]?\d{3}[-\s]?\d{4}|\d{10,12})")
        company_idx = None
        for i, span in enumerate(spans):
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                txt = span.get("text") or ""
                if txt and not re.search(r"form 10-|current report|annual report|transition report|securities and exchange commission|united states", txt, re.I):
                    company_idx = i
                    break
        if company_idx is None:
            return []
        for j in range(company_idx, min(len(spans), company_idx + 30)):
            s2 = spans[j]
            if s2.get("page_no") == 1:
                t2 = ((s2.get("text") or "") + " " + (s2.get("text_span") or "")).strip()
                if phone_pat.search(t2):
                    out.append(s2)
        return out
    except Exception:
        return []
