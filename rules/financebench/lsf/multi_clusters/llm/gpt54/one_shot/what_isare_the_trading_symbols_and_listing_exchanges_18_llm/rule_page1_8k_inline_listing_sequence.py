def rule_page1_8k_inline_listing_sequence(doc: dict) -> list[dict]:
    """Match 8-K cover spans where title, symbol, and exchange appear inline in one large span."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if (
                "securities registered pursuant to section 12(b)" in txt
                and "trading symbol" in txt
                and ("nasdaq" in txt or "stock exchange" in txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
