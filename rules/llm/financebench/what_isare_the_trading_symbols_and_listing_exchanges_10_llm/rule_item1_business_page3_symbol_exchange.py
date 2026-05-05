def rule_item1_business_page3_symbol_exchange(doc: dict) -> list[dict]:
    """Match page 3 Item 1 Business spans, a common alternate location for symbol/exchange statements."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") not in {3, 4, 5, 6}:
                continue
            path = ((span.get("structure", {}) or {}).get("path_text") or "").lower()
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if "item 1" in path and "business" in path and (
                "symbol" in txt or "trades on" in txt or "listed on" in txt or "nasdaq" in txt or "new york stock exchange" in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
