def rule_tables_on_same_page_as_balance_sheet_heading(doc: dict) -> list[dict]:
    """Match tables on pages that also contain a balance-sheet-related heading/text span."""
    try:
        pages = set()
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            if "balance sheet" in txt or "statement of financial position" in txt:
                pages.add(span.get("page_no"))
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table" and span.get("page_no") in pages:
                out.append(span)
        return out
    except Exception:
        return []
