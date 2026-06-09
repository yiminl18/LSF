def rule_short_docs_without_exhibit_section_tail(doc: dict) -> list[dict]:
    """Match the tail of short filings that contain no exhibit section so the model can answer none listed."""
    try:
        texts = doc.get("texts", [])
        if not texts:
            return []

        max_page = max((span.get("page_no") or 0) for span in texts)
        if max_page > 25:
            return []

        has_exhibit_anchor = False
        for span in texts:
            text = " ".join((span.get("text") or "").split()).lower()
            path = " ".join((((span.get("structure") or {}).get("path_text")) or "").split()).lower()
            blob = f"{path} {text}"
            if (
                "item 9.01" in blob
                or ("item 15" in blob and "exhibit" in blob)
                or ("item 16" in blob and "exhibit" in blob)
                or ("item 6" in blob and "exhibit" in blob)
                or "exhibit index" in blob
                or "index to exhibits" in blob
            ):
                has_exhibit_anchor = True
                break
        if has_exhibit_anchor:
            return []

        hits: list[dict] = []
        for span in texts:
            page_no = span.get("page_no") or 0
            text = " ".join((span.get("text") or "").split())
            if page_no < max(1, max_page - 1):
                continue
            if span.get("label") not in {"text", "section_header", "list_item"}:
                continue
            if not text or len(text) > 260:
                continue
            hits.append(span)
        return hits[-6:]
    except Exception:
        return []
