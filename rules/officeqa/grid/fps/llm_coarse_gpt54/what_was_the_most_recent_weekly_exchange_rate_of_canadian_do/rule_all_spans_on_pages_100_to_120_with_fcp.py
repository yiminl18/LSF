def rule_all_spans_on_pages_100_to_120_with_fcp(doc: dict) -> list[dict]:
    """Match spans in the common answer zone pages 100-120 that mention FCP or Canadian positions."""
    try:
        out = []
        for span in doc.get("texts", []):
            page = span.get("page_no")
            txt = (span.get("text") or "").lower()
            if isinstance(page, int) and 100 <= page <= 120:
                if "fcp-" in txt or "canadian dollar positions" in txt or "weekly bank positions" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
