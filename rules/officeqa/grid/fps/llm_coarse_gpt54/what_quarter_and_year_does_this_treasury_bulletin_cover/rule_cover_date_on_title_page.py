def rule_cover_date_on_title_page(doc: dict) -> list[dict]:
    """Match date-like spans on pages containing a Treasury Bulletin title/header."""
    import re
    try:
        texts = doc.get("texts", [])
        title_pages = set()
        for s in texts:
            txt = (s.get("text") or "").strip()
            if re.search(r"treasury\s+bulletin", txt, re.I):
                title_pages.add(s.get("page_no"))
        out = []
        pat = re.compile(
            r"\b("
            r"January|February|March|April|May|June|July|August|September|October|November|December|"
            r"Spring|Summer|Fall|Winter"
            r")\b"
            r".{0,60}?(?:\b\d{4}\b|\bFiscal\s+\d{4}\b)?",
            re.I,
        )
        for s in texts:
            if s.get("page_no") in title_pages:
                txt = (s.get("text") or "").strip()
                if txt and pat.search(txt):
                    out.append(s)
        return out
    except Exception:
        return []
