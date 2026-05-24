def rule_exact_phone_only_span(doc: dict) -> list[dict]:
    """Match spans whose text is primarily just a phone number."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            t = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.fullmatch(r"(\+\d{1,3}\s*\d[\d\s\-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\d{3}-\d{3}-\d{4})", t):
                out.append(span)
        return out
    except Exception:
        return []
