def rule_h2_phone_header(doc: dict) -> list[dict]:
    """Match H2 section headers whose text is itself a phone number."""
    try:
        import re
        out = []
        phone_re = re.compile(r"^\s*(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|[0-9]{3}[-/][0-9]{3}[-/][0-9]{4})\s*$")
        for span in doc.get("texts", []):
            lvl = (span.get("structure", {}) or {}).get("level")
            if lvl == "H2" and phone_re.search(span.get("text", "") or ""):
                out.append(span)
        return out
    except Exception:
        return []
