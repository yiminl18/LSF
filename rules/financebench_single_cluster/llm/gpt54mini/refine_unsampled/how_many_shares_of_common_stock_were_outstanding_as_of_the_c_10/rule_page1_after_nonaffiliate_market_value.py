def rule_page1_after_nonaffiliate_market_value(doc: dict) -> list[dict]:
    """Match page-1 spans with outstanding-share language that follow a non-affiliate market value sentence."""
    out = []
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            t = (span.get("text") or "").lower()
            if span.get("page_no") != 1:
                continue
            if "outstanding" in t and ("common stock" in t or "shares" in t):
                prev = " ".join((p.get("text") or "") for p in texts[max(0, i - 6):i]).lower()
                if "non-affiliates" in prev or "held by non-affiliates" in prev or "aggregate market value" in prev:
                    out.append(span)
    except Exception:
        return []
    return out
