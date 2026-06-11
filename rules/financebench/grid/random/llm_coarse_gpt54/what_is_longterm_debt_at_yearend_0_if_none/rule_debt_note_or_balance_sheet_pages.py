def rule_debt_note_or_balance_sheet_pages(doc: dict) -> list[dict]:
    """Match tables on pages likely to contain either debt notes or balance sheets if they mention debt."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            page = span.get("page_no")
            if page is None:
                continue
            if span.get("label") != "table":
                continue
            if not (3 <= page <= 120):
                continue
            text = span.get("text", "") or ""
            if re.search(r"\bdebt\b|\blong[\-\s]?term debt\b|\bborrowings\b", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
