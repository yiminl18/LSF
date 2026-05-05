def rule_phone_in_path_text(doc: dict) -> list[dict]:
    """Match spans whose path_text itself contains a phone number or telephone label."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure", {}) or {}).get("path_text") or "").strip()
            if span.get("page_no") == 1 and re.search(r"telephone|area code|(\(\d{3}\)\s*\d{3}[-\s]?\d{4})|(\+\d{1,3}\s*\d)|\b\d{3}-\d{3}-\d{4}\b", path, re.I):
                out.append(span)
        return out
    except Exception:
        return []
