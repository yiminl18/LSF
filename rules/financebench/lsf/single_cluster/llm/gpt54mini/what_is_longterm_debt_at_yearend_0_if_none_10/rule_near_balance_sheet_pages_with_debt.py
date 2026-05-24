def rule_near_balance_sheet_pages_with_debt(doc: dict) -> list[dict]:
    """Match tables near TOC-inferred balance sheet pages that mention debt."""
    import re
    try:
        out = []
        for span in rule_near_balance_sheet_pages_from_toc(doc):
            txt = (span.get("text") or "").lower()
            if re.search(r"\bdebt\b", txt) or re.search(r"\bborrowings\b", txt):
                out.append(span)
        return out
    except Exception:
        return []
