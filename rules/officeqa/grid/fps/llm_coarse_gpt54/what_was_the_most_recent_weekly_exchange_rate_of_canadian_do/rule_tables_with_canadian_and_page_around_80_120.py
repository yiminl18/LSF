def rule_tables_with_canadian_and_page_around_80_120(doc: dict) -> list[dict]:
    """Match Canadian dollar position tables on the typical page range where the answer appears."""
    try:
        out = []
        for span in doc.get("texts", []):
            page = span.get("page_no")
            txt = (span.get("text") or "").lower()
            if span.get("label") == "table" and isinstance(page, int) and 80 <= page <= 120:
                if "canadian dollar positions" in txt or "fcp-ii-" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
