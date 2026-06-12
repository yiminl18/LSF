def rule_top_of_document_month_year(doc: dict) -> list[dict]:
    """Match month/year or season/year spans appearing very early in the document."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(
            r"^(?:\s*)("
            r"January|February|March|April|May|June|July|August|September|October|November|December|"
            r"Spring|Summer|Fall|Winter"
            r")(?:\s+Issue)?(?:\s+|,\s*|\s+of\s+)?(?:[A-Za-z]+\s+Quarter,?\s+Fiscal\s+\d{4}|\d{4}|Fiscal\s+\d{4})?",
            re.I,
        )
        out = []
        for s in texts[:120]:
            txt = (s.get("text") or "").strip()
            if txt and pat.search(txt):
                out.append(s)
        return out
    except Exception:
        return []
