def rule_page_with_balance_sheet_header(doc: dict) -> list[dict]:
    """Match all spans on pages containing a balance sheet header."""
    pages = set()
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            if "balance sheet" in txt or "balance sheets" in txt:
                pages.add(span.get("page_no"))
        for span in doc.get("texts", []):
            if span.get("page_no") in pages:
                out.append(span)
    except Exception:
        return []
    return out
