def rule_page1_depth_le4_phone(doc: dict) -> list[dict]:
    """Match phone-number spans on page 1 with depth at most 4."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{3}[-/]\d{3}[-/]\d{4})")
        for span in doc.get("texts", []):
            depth = (span.get("structure", {}) or {}).get("depth", 99)
            if span.get("page_no") == 1 and depth <= 4 and phone_re.search(span.get("text", "") or ""):
                out.append(span)
        return out
    except Exception:
        return []
