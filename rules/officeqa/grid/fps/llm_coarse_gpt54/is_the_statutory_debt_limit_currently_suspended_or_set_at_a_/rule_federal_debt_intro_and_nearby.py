def rule_federal_debt_intro_and_nearby(doc: dict) -> list[dict]:
    """Match spans on pages where a Federal Debt introduction/header appears, to capture nearby answer tables."""
    try:
        pages = set()
        for span in doc.get("texts", []):
            text = (span.get("text") or "").lower()
            if "federal debt" in text and span.get("label") in {"section_header", "text"}:
                pages.add(span.get("page_no"))
        out = []
        for span in doc.get("texts", []):
            p = span.get("page_no")
            if p in pages or (isinstance(p, int) and (p - 1 in pages or p + 1 in pages)):
                out.append(span)
        return out
    except Exception:
        return []
