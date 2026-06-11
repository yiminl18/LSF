def rule_phone_number_with_registrant_label_same_span(doc: dict) -> list[dict]:
    """Match spans where a phone number and registrant telephone label co-occur."""
    try:
        import re
        out = []
        phone_pat = re.compile(r"(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]\d{3}[-\s]\d{4}|\d{10,12})")
        for span in doc.get("texts", []):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r"registrant[’'`s]{0,2}\s+telephone\s+number", txt, re.I) and phone_pat.search(txt):
                out.append(span)
        return out
    except Exception:
        return []
