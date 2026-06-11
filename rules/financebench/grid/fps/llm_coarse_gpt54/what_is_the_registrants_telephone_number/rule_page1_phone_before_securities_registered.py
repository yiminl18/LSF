def rule_page1_phone_before_securities_registered(doc: dict) -> list[dict]:
    """Match spans where a phone number appears near 'Securities registered pursuant to Section 12(b)'."""
    try:
        import re
        out = []
        pat = re.compile(
            r"(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]?\d{3}[-\s]?\d{4}|\d{10,12}).{0,200}securities registered pursuant to section 12\(b\)",
            re.I | re.S
        )
        for span in doc.get("texts", []):
            if span.get("page_no") == 1:
                txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
                if pat.search(txt):
                    out.append(span)
        return out
    except Exception:
        return []
