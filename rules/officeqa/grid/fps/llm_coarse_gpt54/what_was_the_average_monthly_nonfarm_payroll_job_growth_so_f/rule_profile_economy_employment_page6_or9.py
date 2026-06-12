def rule_profile_economy_employment_page6_or9(doc: dict) -> list[dict]:
    """Match likely answer spans on the most common answer pages 6 or 9 with labor/payroll wording."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "text":
                continue
            if span.get("page_no") not in (6, 9):
                continue
            txt = span.get("text") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            if (
                "Profile of the Economy" in path
                and re.search(r'(payroll|job growth|employment|unemployment)', txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
