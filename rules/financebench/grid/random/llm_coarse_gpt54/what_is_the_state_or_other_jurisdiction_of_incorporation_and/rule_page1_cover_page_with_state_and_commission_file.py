def rule_page1_cover_page_with_state_and_commission_file(doc: dict) -> list[dict]:
    """Match page-1 spans containing a state/jurisdiction value and commission file wording."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"Commission File", txt, re.I) and re.search(r"\bDelaware\b|\bWashington\b|\bNew York\b|\bJersey\b", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
