def rule_page1_us_phone_without_parentheses(doc: dict) -> list[dict]:
    """Match standalone U.S. phone spans without parenthesized area code on page 1."""
    try:
        import re
        out = []
        us_re = re.compile(r"^\s*\d{3}[-/]\d{3}[-]\d{4}\s*$|^\s*\d{3}[-\s]\d{3}[-]\d{4}\s*$")
        for span in doc.get("texts", []):
            if span.get("page_no") == 1 and us_re.match((span.get("text", "") or "").strip()):
                out.append(span)
        return out
    except Exception:
        return []
