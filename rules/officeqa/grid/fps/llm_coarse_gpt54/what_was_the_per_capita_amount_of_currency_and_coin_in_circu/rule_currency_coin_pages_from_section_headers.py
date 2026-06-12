def rule_currency_coin_pages_from_section_headers(doc: dict) -> list[dict]:
    """Return all spans on pages where a currency/coin section header appears."""
    import re
    out = []
    try:
        pages = set()
        for span in doc.get("texts", []):
            if span.get("label") == "section_header" and re.search(r'currency\s+and\s+coin', (span.get("text") or ""), re.I):
                pages.add(span.get("page_no"))
        if not pages:
            return []
        for span in doc.get("texts", []):
            if span.get("page_no") in pages:
                out.append(span)
    except Exception:
        return []
    return out
