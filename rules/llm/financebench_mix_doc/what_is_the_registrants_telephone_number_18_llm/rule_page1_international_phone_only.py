def rule_page1_international_phone_only(doc: dict) -> list[dict]:
    """Match standalone international-format phone spans like '+44 117 9753200' on page 1."""
    try:
        import re
        out = []
        intl_re = re.compile(r"^\s*\+\d{1,3}\s+\d{2,4}\s+\d{4,}\s*$")
        for span in doc.get("texts", []):
            if span.get("page_no") == 1 and intl_re.match((span.get("text", "") or "").strip()):
                out.append(span)
        return out
    except Exception:
        return []
