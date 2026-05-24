def rule_exhibit_pages_high_page_numbers(doc: dict) -> list[dict]:
    """Match spans on the highest-numbered pages that mention exhibits, where exhibit indexes often appear."""
    try:
        texts = doc.get("texts", [])
        if not texts:
            return []
        max_page = max((s.get("page_no") or 0) for s in texts)
        candidate_pages = {max_page, max_page - 1, max_page - 2, max_page - 3}
        out = []
        for span in texts:
            txt = (span.get("text") or "")
            path = (((span.get("structure") or {}).get("path_text")) or "")
            if span.get("page_no") in candidate_pages and ("exhibit" in txt.lower() or "exhibit" in path.lower()):
                out.append(span)
        return out
    except Exception:
        return []
