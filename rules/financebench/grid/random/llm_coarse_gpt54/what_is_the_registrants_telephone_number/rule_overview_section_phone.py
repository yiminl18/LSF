def rule_overview_section_phone(doc: dict) -> list[dict]:
    """Match overview-section spans that include the company telephone number."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{3}[-/]\d{3}[-/]\d{4})")
        for span in doc.get("texts", []):
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if re.search(r"Overview|OVERVIEW", path) and phone_re.search(span.get("text", "") or ""):
                out.append(span)
        return out
    except Exception:
        return []
