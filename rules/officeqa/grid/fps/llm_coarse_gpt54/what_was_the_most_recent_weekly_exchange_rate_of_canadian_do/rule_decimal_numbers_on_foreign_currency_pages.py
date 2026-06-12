def rule_decimal_numbers_on_foreign_currency_pages(doc: dict) -> list[dict]:
    """Match decimal-number spans on pages containing foreign currency positions, useful for answer extraction."""
    try:
        import re
        texts = doc.get("texts", [])
        pages = set()
        for span in texts:
            txt = (span.get("text") or "").lower()
            if "foreign currency positions" in txt or "canadian dollar positions" in txt:
                p = span.get("page_no")
                if isinstance(p, int):
                    pages.add(p)
        out = []
        for span in texts:
            if span.get("page_no") in pages:
                stxt = span.get("text") or ""
                if re.search(r"\b\d+\.\d{3,4}\b", stxt):
                    out.append(span)
        return out
    except Exception:
        return []
