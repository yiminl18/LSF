def rule_cover_date_on_early_pages(doc: dict) -> list[dict]:
    """Match month/season + year/date-like spans on the first few pages where the cover date usually appears."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(
            r"\b("
            r"January|February|March|April|May|June|July|August|September|October|November|December|"
            r"Spring|Summer|Fall|Winter"
            r")\b"
            r"(?:\s+Issue)?"
            r"(?:\s+(?:of\s+)?)?"
            r"(?:\s+(?:First|Second|Third|Fourth|1st|2nd|3rd|4th)\s+Quarter,?\s+Fiscal\s+\d{4})?"
            r"(?:\s+\d{4})?",
            re.I,
        )
        out = []
        for span in texts:
            if span.get("page_no", 999) <= 11:
                txt = (span.get("text") or "").strip()
                if txt and pat.search(txt):
                    out.append(span)
        return out
    except Exception:
        return []
