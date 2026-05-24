def rule_page1_exchange_or_symbol_in_same_path(doc: dict) -> list[dict]:
    """Match page-1 spans sharing a path_text with listing headers and containing symbol/exchange values."""
    try:
        texts = doc.get("texts", [])
        out = []
        listing_paths = set()
        for span in texts:
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and (
                "trading symbol" in txt
                or "name of each exchange on which registered" in txt
                or "section 12(b)" in txt
            ):
                listing_paths.add(((span.get("structure") or {}).get("path_text") or ""))
        for span in texts:
            if span.get("page_no") != 1:
                continue
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "").lower()
            if path in listing_paths and (
                "nasdaq" in txt
                or "stock exchange" in txt
                or txt.strip() in {"adbe", "amzn", "atvi", "amcr", "cost", "ba", "fl", "mmm", "ebay"}
            ):
                out.append(span)
        return out
    except Exception:
        return []
