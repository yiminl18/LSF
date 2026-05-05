def rule_balance_sheet_table_on_pages_30_to_70(doc: dict) -> list[dict]:
    """Match mid-document financial statement tables on typical balance-sheet pages that mention long-term debt."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            page = span.get("page_no")
            if page is None or not (30 <= page <= 70):
                continue
            txt = (span.get("text") or "").lower()
            if "balance sheet" in txt and re.search(r"\blong[\-\s]?term debt\b", txt):
                out.append(span)
        return out
    except Exception:
        return []
