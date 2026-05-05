def rule_page1_top_half(doc: dict) -> list[dict]:
    """Match all spans on the first page in the top half of the document order."""
    try:
        texts = doc.get("texts", [])
        page1 = [s for s in texts if s.get("page_no") == 1]
        cutoff = max(1, len(page1) // 2)
        return page1[:cutoff]
    except Exception:
        return []
