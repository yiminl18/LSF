def rule_tables_with_canadian_on_late_pages(doc: dict) -> list[dict]:
    """Match Canadian dollar position tables on later statistical pages, excluding contents/front matter."""
    try:
        out = []
        for span in doc.get("texts", []):
            page = span.get("page_no")
            txt = (span.get("text") or "").lower()
            if span.get("label") == "table" and isinstance(page, int) and page >= 70:
                if "canadian dollar positions" in txt or "fcp-ii-" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
