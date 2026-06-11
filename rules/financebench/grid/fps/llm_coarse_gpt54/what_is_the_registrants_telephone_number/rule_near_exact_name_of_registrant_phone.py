def rule_near_exact_name_of_registrant_phone(doc: dict) -> list[dict]:
    """Match phone-like spans near the registrant name block on the cover page."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        phone_pat = re.compile(r"(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]?\d{3}[-\s]?\d{4}|\d{10,12})")
        for i, span in enumerate(spans):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r"exact name of registrant", txt, re.I):
                for j in range(max(0, i - 5), min(len(spans), i + 15)):
                    s2 = spans[j]
                    if s2.get("page_no") == span.get("page_no") == 1:
                        t2 = ((s2.get("text") or "") + " " + (s2.get("text_span") or "")).strip()
                        if phone_pat.search(t2):
                            out.append(s2)
        return out
    except Exception:
        return []
