def rule_page1_adobe_sanjose_header(doc: dict) -> list[dict]:
    """Match Adobe-style page-1 address line with San Jose, California."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.search(r'345 Park Avenue,\s*San Jose,\s*California\s*95110', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
