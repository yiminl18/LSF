def rule_page1_sentence_with_market_value_and_outstanding_same_block(doc: dict) -> list[dict]:
    """Match page-1 cover blocks containing both market-value and outstanding-share disclosures."""
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            full = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if "aggregate market value" in full and "outstanding" in full and ("common stock" in full or "shares" in full):
                out.append(span)
    except Exception:
        return []
    return out
