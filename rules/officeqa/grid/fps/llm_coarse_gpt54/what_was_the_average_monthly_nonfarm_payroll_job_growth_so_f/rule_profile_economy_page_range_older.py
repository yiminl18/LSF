def rule_profile_economy_page_range_older(doc: dict) -> list[dict]:
    """Match likely answer spans on pages 5-7 in older issues under Profile of the Economy with labor keywords."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            page = span.get("page_no")
            txt = span.get("text") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            if (
                span.get("label") == "text"
                and isinstance(page, int) and 5 <= page <= 7
                and "Profile of the Economy" in path
                and re.search(r'(labor|employment|unemployment|payroll|job growth)', txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
