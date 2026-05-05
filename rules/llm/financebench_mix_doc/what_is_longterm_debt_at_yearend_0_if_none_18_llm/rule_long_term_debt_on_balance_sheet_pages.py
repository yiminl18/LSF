def rule_long_term_debt_on_balance_sheet_pages(doc: dict) -> list[dict]:
    """Match spans mentioning long-term debt on pages that also contain balance sheet content."""
    import re
    pages = set()
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            if "balance sheet" in txt or "balance sheets" in txt:
                pages.add(span.get("page_no"))
        for span in doc.get("texts", []):
            if span.get("page_no") in pages and re.search(r"\blong[- ]term debt\b|\bdebt\b", span.get("text") or "", re.I):
                out.append(span)
    except Exception:
        return []
    return out
