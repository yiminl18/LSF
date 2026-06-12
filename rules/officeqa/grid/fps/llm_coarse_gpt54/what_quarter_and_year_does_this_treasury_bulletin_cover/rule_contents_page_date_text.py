def rule_contents_page_date_text(doc: dict) -> list[dict]:
    """Match date-like text on pages whose main section is Contents."""
    import re
    try:
        texts = doc.get("texts", [])
        contents_pages = set()
        for s in texts:
            txt = (s.get("text") or "").strip()
            if txt.lower() in {"contents", "table of contents"}:
                contents_pages.add(s.get("page_no"))
        pat = re.compile(
            r"\b("
            r"January|February|March|April|May|June|July|August|September|October|November|December|"
            r"Spring|Summer|Fall|Winter|"
            r"First Quarter, Fiscal \d{4}|Second Quarter, Fiscal \d{4}|Third Quarter, Fiscal \d{4}|Fourth Quarter, Fiscal \d{4}"
            r")\b",
            re.I,
        )
        return [
            s for s in texts
            if s.get("page_no") in contents_pages and pat.search((s.get("text") or "").strip())
        ]
    except Exception:
        return []
