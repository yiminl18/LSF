def rule_balance_sheet_long_term_debt_page_range(doc: dict) -> list[dict]:
    """Match table spans on financial statement pages likely to contain long-term debt."""
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            page = span.get("page_no", 0)
            text = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if 3 <= page <= 80 and (
                "balance sheet" in text
                or "balance sheets" in text
                or "financial statements" in path
                or "item 1. condensed consolidated financial statements" in path
            ):
                out.append(span)
    except Exception:
        return []
    return out
