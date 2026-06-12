def rule_page_header_date(doc: dict) -> list[dict]:
    """Match page_header spans that contain the issue month/season and often repeat the bulletin date."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(
            r"\b("
            r"January|February|March|April|May|June|July|August|September|October|November|December|"
            r"Spring|Summer|Fall|Winter"
            r")\b"
            r".{0,50}?(?:\b\d{4}\b|\bFiscal\s+\d{4}\b)?",
            re.I,
        )
        return [
            s for s in texts
            if s.get("label") == "page_header" and pat.search((s.get("text") or "").strip())
        ]
    except Exception:
        return []
