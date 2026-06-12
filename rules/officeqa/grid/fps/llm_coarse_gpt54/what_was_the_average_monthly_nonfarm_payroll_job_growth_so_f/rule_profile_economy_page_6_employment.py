def rule_profile_economy_page_6_employment(doc: dict) -> list[dict]:
    """Match likely answer spans on page 6 in older bulletins under Profile of the Economy employment discussion."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            if (
                span.get("page_no") == 6
                and span.get("label") == "text"
                and "Profile of the Economy" in path
                and re.search(r'(nonfarm payroll|payroll job|Employment and unemployment)', txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
