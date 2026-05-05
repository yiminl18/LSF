def rule_non_debt_sec_cover_only(doc: dict) -> list[dict]:
    """Match cover-page only documents with no financial statement sections, useful for 0 debt cases."""
    out = []
    try:
        has_financials = False
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "financial statements" in txt or "balance sheet" in txt or "condensed consolidated" in txt or "item 8" in path:
                has_financials = True
                break
        if not has_financials:
            for span in doc.get("texts", []):
                if span.get("page_no") == 1:
                    out.append(span)
    except Exception:
        return []
    return out
