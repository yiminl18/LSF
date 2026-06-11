def rule_phone_number_after_registrant_label_same_span(doc: dict) -> list[dict]:
    """Match spans where the registrant telephone label appears before a phone number."""
    try:
        import re
        out = []
        pat = re.compile(
            r"registrant[’'`s]{0,2}\s+telephone\s+number.{0,120}(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]\d{3}[-\s]\d{4}|\d{10,12})",
            re.I | re.S
        )
        for span in doc.get("texts", []):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if pat.search(txt):
                out.append(span)
        return out
    except Exception:
        return []
