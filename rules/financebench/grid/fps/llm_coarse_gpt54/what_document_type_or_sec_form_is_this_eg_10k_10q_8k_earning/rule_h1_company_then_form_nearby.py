def rule_h1_company_then_form_nearby(doc: dict) -> list[dict]:
    """Match form headings on page 1 that appear before or near the first company-name H1/header."""
    import re
    try:
        texts = doc.get("texts", [])
        company_idx = None
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                txt = (span.get("text") or "").strip()
                if txt and not re.match(r"^(FORM|CURRENT REPORT|NEWS RELEASE|UNITED STATES|SECURITIES AND EXCHANGE COMMISSION|WASHINGTON)", txt, re.I):
                    company_idx = i
                    break
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.match(r"^FORM\s+[A-Z0-9\-]+", txt, re.I):
                if company_idx is None or i <= company_idx:
                    out.append(span)
        return out
    except Exception:
        return []
