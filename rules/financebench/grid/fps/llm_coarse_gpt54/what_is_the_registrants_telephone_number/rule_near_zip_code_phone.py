def rule_near_zip_code_phone(doc: dict) -> list[dict]:
    """Match phone-like spans near a zip-code label on the cover page."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        phone_pat = re.compile(r"(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]?\d{3}[-\s]?\d{4}|\d{10,12})")
        for i, span in enumerate(spans):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r"\(zip code\)|zip code", txt, re.I):
                for j in range(max(0, i - 3), min(len(spans), i + 10)):
                    s2 = spans[j]
                    if s2.get("page_no") == 1:
                        t2 = ((s2.get("text") or "") + " " + (s2.get("text_span") or "")).strip()
                        if phone_pat.search(t2):
                            out.append(s2)
        return out
    except Exception:
        return []
