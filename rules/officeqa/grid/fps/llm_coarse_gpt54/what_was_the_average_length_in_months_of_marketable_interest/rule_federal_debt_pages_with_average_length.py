def rule_federal_debt_pages_with_average_length(doc: dict) -> list[dict]:
    """Return all spans on pages containing a Federal Debt target heading mentioning average length."""
    out = []
    try:
        target_pages = set()
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "average length" in txt and ("federal debt" in path or "debt" in txt):
                if span.get("page_no") is not None:
                    target_pages.add(span.get("page_no"))
        for span in doc.get("texts", []):
            if span.get("page_no") in target_pages:
                out.append(span)
    except Exception:
        return []
    return out
