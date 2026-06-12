def rule_page_header_or_footer_repeated_issue_date(doc: dict) -> list[dict]:
    """Match repeated issue date spans in page headers/footers across the document."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(
            r"\b("
            r"January|February|March|April|May|June|July|August|September|October|November|December|"
            r"Spring|Summer|Fall|Winter"
            r")\b.*?(?:\b\d{4}\b|\bFiscal\s+\d{4}\b)?",
            re.I,
        )
        return [
            s for s in texts
            if s.get("label") in {"page_header", "page_footer"} and pat.search((s.get("text") or "").strip())
        ]
    except Exception:
        return []
