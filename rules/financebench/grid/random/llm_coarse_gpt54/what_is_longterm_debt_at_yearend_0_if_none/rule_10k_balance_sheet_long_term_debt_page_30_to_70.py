def rule_10k_balance_sheet_long_term_debt_page_30_to_70(doc: dict) -> list[dict]:
    """Match 10-K balance sheet tables on later financial statement pages with long-term debt rows."""
    try:
        import re
        out = []
        is_10k = any(re.search(r"form 10-k", (s.get("text", "") or ""), re.I) for s in doc.get("texts", []))
        if not is_10k:
            return []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            page = span.get("page_no")
            if page is None or not (30 <= page <= 80):
                continue
            text = span.get("text", "") or ""
            path = (span.get("structure") or {}).get("path_text", "") or ""
            if re.search(r"balance sheet", text + " " + path, re.I) and re.search(r"\blong[\-\s]?term debt\b|\bdebt\b|\bborrowings\b", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
