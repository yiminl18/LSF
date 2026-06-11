def rule_page1_phone_near_commission_file_number(doc: dict) -> list[dict]:
    """Match phone-like spans near a commission file number on page 1."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        phone_pat = re.compile(r"(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]?\d{3}[-\s]?\d{4}|\d{10,12})")
        for i, span in enumerate(spans):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r"commission file number|commission file no|file number", txt, re.I):
                for j in range(i, min(len(spans), i + 20)):
                    s2 = spans[j]
                    if s2.get("page_no") == 1:
                        t2 = ((s2.get("text") or "") + " " + (s2.get("text_span") or "")).strip()
                        if phone_pat.search(t2):
                            out.append(s2)
        return out
    except Exception:
        return []
