def rule_canadian_positions_pages(doc: dict) -> list[dict]:
    """Match all spans on pages that contain a CANADIAN DOLLAR POSITIONS mention."""
    try:
        texts = doc.get("texts", [])
        pages = set()
        for span in texts:
            txt = (span.get("text") or "").lower()
            if "canadian dollar positions" in txt:
                p = span.get("page_no")
                if isinstance(p, int):
                    pages.add(p)
        return [s for s in texts if s.get("page_no") in pages]
    except Exception:
        return []
