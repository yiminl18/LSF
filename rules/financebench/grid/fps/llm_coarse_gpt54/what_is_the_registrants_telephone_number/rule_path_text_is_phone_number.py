def rule_path_text_is_phone_number(doc: dict) -> list[dict]:
    """Match spans whose structure.path_text ends with or contains a phone-number-like heading."""
    try:
        import re
        out = []
        pat = re.compile(r"(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]?\d{3}[-\s]?\d{4}|\d{10,12})")
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            if pat.search(path):
                out.append(span)
        return out
    except Exception:
        return []
