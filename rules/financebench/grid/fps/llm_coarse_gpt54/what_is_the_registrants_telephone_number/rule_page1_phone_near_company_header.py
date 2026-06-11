def rule_page1_phone_near_company_header(doc: dict) -> list[dict]:
    """Match phone-like spans within a short window after the main company header on page 1."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        phone_pat = re.compile(r"(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]?\d{3}[-\s]?\d{4}|\d{10,12})")
        for i, span in enumerate(spans):
            if span.get("page_no") != 1:
                continue
            if span.get("label") == "section_header":
                txt = span.get("text") or ""
                if txt and not re.search(r"form 10-|current report|annual report|transition report|securities and exchange commission", txt, re.I):
                    for j in range(i, min(len(spans), i + 20)):
                        s2 = spans[j]
                        if s2.get("page_no") == 1:
                            t2 = ((s2.get("text") or "") + " " + (s2.get("text_span") or "")).strip()
                            if phone_pat.search(t2):
                                out.append(s2)
        return out
    except Exception:
        return []
