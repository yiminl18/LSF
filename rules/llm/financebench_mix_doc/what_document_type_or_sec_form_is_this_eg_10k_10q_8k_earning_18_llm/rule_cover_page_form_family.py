def rule_cover_page_form_family(doc: dict) -> list[dict]:
    """Match all cover-page spans belonging to the form family: form heading plus annual/quarterly/current report phrases."""
    import re
    try:
        out = []
        for s in doc.get("texts", []):
            if s.get("page_no") != 1:
                continue
            txt = ((s.get("text") or "") + " " + (s.get("text_span") or "")).upper()
            if (
                re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", txt, re.I)
                or "CURRENT REPORT" in txt
                or "ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(D)" in txt
                or "QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(D)" in txt
                or "TRANSITION REPORT PURSUANT TO SECTION 13 OR 15(D)" in txt
            ):
                out.append(s)
        return out
    except Exception:
        return []
