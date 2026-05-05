def rule_page1_uk_address_with_bristol(doc: dict) -> list[dict]:
    """Match page-1 spans containing the UK Bristol address pattern."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r'Warmley.*Bristol.*BS30 8XP.*United Kingdom|83 Tower Road North.*Bristol', text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
