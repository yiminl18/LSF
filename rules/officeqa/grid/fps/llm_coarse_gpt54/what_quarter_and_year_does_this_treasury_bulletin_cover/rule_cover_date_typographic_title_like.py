def rule_cover_date_typographic_title_like(doc: dict) -> list[dict]:
    """Match large/title-like spans on early pages that look like the issue date."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(
            r"(January|February|March|April|May|June|July|August|September|October|November|December|Spring|Summer|Fall|Winter)",
            re.I,
        )
        out = []
        for s in texts:
            if s.get("page_no", 999) <= 9:
                level = ((s.get("structure") or {}).get("level") or "")
                txt = (s.get("text") or "").strip()
                if txt and pat.search(txt) and (level in {"H1", "H2"} or s.get("bold") == 1):
                    out.append(s)
        return out
    except Exception:
        return []
