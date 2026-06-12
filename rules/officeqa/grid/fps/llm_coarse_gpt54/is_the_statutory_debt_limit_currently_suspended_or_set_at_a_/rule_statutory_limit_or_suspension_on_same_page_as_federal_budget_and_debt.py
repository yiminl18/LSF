def rule_statutory_limit_or_suspension_on_same_page_as_federal_budget_and_debt(doc: dict) -> list[dict]:
    """Return spans on pages containing the Federal Budget and Debt header that mention limit, ceiling, or suspension."""
    import re
    try:
        pages = set()
        for span in doc.get("texts", []):
            if (span.get("text") or "").strip().lower() == "federal budget and debt":
                pages.add(span.get("page_no"))
        if not pages:
            return []
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") in pages:
                text = (span.get("text") or "")
                if re.search(r"debt|ceiling|limit|suspend", text, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
