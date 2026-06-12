def rule_page_range_main_financial_tables(doc: dict) -> list[dict]:
    """Match likely answer tables on early financial-operations pages (roughly pages 15-25 in older issues, 15-20 in later issues)."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            p = span.get("page_no")
            txt = span.get("text") or ""
            if isinstance(p, int) and 14 <= p <= 25 and (
                re.search(r'Summary of Fiscal Operations', txt, re.I)
                or re.search(r'Budget Receipts by Source', txt, re.I)
                or re.search(r'Fiscal \d{4} to date', txt, re.I)
                or re.search(r'Actual fiscal year to date', txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
