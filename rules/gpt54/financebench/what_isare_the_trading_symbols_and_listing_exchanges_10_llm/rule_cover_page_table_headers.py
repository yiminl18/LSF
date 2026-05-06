def rule_cover_page_table_headers(doc: dict) -> list[dict]:
    """Match page 1 spans that are cover-page headers for security registration fields."""
    try:
        texts = doc.get("texts", [])
        out = []
        headers = {
            "title of each class",
            "trading symbol",
            "trading symbol(s)",
            "name of each exchange on which registered",
            "name of exchange on which registered",
            "name of each exchange",
        }
        for span in texts:
            txt = (span.get("text") or "").strip().lower()
            if span.get("page_no") == 1 and txt in headers:
                out.append(span)
        return out
    except Exception:
        return []
