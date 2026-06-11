def rule_10q_balance_sheet_long_term_debt_page_3_to_8(doc: dict) -> list[dict]:
    """Match 10-Q balance sheet tables on early pages with long-term debt rows."""
    try:
        import re
        out = []
        is_10q = any(re.search(r"form 10-q", (s.get("text", "") or ""), re.I) for s in doc.get("texts", []))
        if not is_10q:
            return []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            page = span.get("page_no")
            if page is None or not (3 <= page <= 8):
                continue
            text = span.get("text", "") or ""
            if re.search(r"balance sheet", text, re.I) and re.search(r"\blong[\-\s]?term debt\b|\bdebt\b", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
