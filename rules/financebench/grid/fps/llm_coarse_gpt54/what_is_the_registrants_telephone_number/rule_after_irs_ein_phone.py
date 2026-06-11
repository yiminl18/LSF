def rule_after_irs_ein_phone(doc: dict) -> list[dict]:
    """Match phone-like spans shortly after the IRS Employer Identification label."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        phone_pat = re.compile(r"(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]?\d{3}[-\s]?\d{4}|\d{10,12})")
        for i, span in enumerate(spans):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r"i\.?r\.?s\.?\s+employer identification|employer identification no", txt, re.I):
                for j in range(i, min(len(spans), i + 12)):
                    s2 = spans[j]
                    if s2.get("page_no") == 1:
                        t2 = ((s2.get("text") or "") + " " + (s2.get("text_span") or "")).strip()
                        if phone_pat.search(t2):
                            out.append(s2)
        return out
    except Exception:
        return []
