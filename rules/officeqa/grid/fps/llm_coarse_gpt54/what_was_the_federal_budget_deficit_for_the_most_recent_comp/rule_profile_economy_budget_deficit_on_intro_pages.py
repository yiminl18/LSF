def rule_profile_economy_budget_deficit_on_intro_pages(doc: dict) -> list[dict]:
    """Match likely answer spans on the first Profile of the Economy article pages."""
    out = []
    try:
        for span in doc.get("texts", []):
            page = span.get("page_no")
            txt = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if (
                isinstance(page, int) and 5 <= page <= 12
                and span.get("label") == "text"
                and "profile of the economy" in path
                and ("budget deficit" in txt or "federal deficit" in txt)
            ):
                out.append(span)
    except Exception:
        return []
    return out
