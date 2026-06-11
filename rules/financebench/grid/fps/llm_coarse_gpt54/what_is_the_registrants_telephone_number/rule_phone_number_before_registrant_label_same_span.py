def rule_phone_number_before_registrant_label_same_span(doc: dict) -> list[dict]:
    """Match spans where a phone number appears before the registrant telephone label."""
    try:
        import re
        out = []
        pat = re.compile(
            r"(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]\d{3}[-\s]\d{4}|\d{10,12}).{0,120}registrant[’'`s]{0,2}\s+telephone\s+number",
            re.I | re.S
        )
        for span in doc.get("texts", []):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if pat.search(txt):
                out.append(span)
        return out
    except Exception:
        return []
