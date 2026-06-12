def rule_cover_page_centerpiece_date(doc: dict) -> list[dict]:
    """Match standalone date-like text on the main cover page near Treasury Bulletin title."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(
            r"^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}$|"
            r"^(Spring|Summer|Fall|Winter)\s+Issue$|"
            r"^(Fall|Winter|Spring|Summer)\s+Issue(?:\s+of\s+(?:First|Second|Third|Fourth)\s+Quarter,?\s+Fiscal\s+\d{4})?$|"
            r"^(FIRST|SECOND|THIRD|FOURTH)\s+QUARTER,?\s+FISCAL\s+\d{4}$",
            re.I,
        )
        out = []
        for s in texts:
            if s.get("page_no", 999) <= 9:
                txt = (s.get("text") or "").strip()
                if txt and pat.search(txt):
                    out.append(s)
        return out
    except Exception:
        return []
