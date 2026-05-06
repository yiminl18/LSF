def rule_page1_known_company_cover_listing_block(doc: dict) -> list[dict]:
    """Match page-1 spans in the cover block that mention both a security class and listing metadata."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if span.get("page_no") == 1 and (
                "common stock" in txt or "ordinary shares" in txt or "notes due" in txt
            ):
                if (
                    "trading symbol" in txt
                    or "exchange" in txt
                    or "section 12(b)" in txt
                    or "nasdaq" in txt
                    or "stock exchange" in txt
                ):
                    out.append(span)
        return out
    except Exception:
        return []
