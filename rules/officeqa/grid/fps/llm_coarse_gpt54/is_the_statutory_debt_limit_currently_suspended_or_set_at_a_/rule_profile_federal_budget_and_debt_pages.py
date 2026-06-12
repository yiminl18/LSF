def rule_profile_federal_budget_and_debt_pages(doc: dict) -> list[dict]:
    """Match spans on pages containing the modern Profile of the Economy discussion of Federal Budget and Debt."""
    try:
        pages = set()
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip().lower()
            if text == "federal budget and debt":
                pages.add(span.get("page_no"))
        if not pages:
            return []
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") in pages:
                out.append(span)
        return out
    except Exception:
        return []
