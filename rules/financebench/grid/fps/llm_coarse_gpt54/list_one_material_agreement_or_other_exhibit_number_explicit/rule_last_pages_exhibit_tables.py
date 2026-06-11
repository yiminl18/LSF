def rule_last_pages_exhibit_tables(doc: dict) -> list[dict]:
    """Match exhibit tables on the last few pages, where exhibit indexes usually appear."""
    out = []
    try:
        texts = doc.get("texts", [])
        if not texts:
            return []
        max_page = max((s.get("page_no") or 0) for s in texts)
        for span in texts:
            if span.get("label") == "table" and (span.get("page_no") or 0) >= max_page - 3:
                text = span.get("text", "") or ""
                if "Exhibit" in text or "Description" in text:
                    out.append(span)
    except Exception:
        return []
    return out
