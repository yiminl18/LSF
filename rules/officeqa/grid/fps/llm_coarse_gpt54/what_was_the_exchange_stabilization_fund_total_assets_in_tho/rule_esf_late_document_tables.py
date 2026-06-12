def rule_esf_late_document_tables(doc: dict) -> list[dict]:
    """Match tables in the latter half of the document, where ESF often appears."""
    try:
        texts = doc.get("texts", [])
        if not texts:
            return []
        pages = [s.get("page_no") for s in texts if isinstance(s.get("page_no"), int)]
        if not pages:
            return []
        cutoff = (min(pages) + max(pages)) / 2.0
        return [
            s for s in texts
            if s.get("label") == "table"
            and isinstance(s.get("page_no"), int)
            and s.get("page_no") >= cutoff
        ]
    except Exception:
        return []
