def rule_page1_after_market_value_before_documents(doc: dict) -> list[dict]:
    """Match page-1 spans with outstanding-share language occurring after market-value disclosure and before documents-incorporated text."""
    out = []
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            text = (span.get("text") or "").lower()
            if span.get("page_no") != 1:
                continue
            if "outstanding" not in text:
                continue
            if "common stock" not in text and "shares" not in text:
                continue
            prev_window = texts[max(0, i - 5):i]
            next_window = texts[i + 1:i + 6]
            has_market = any("aggregate market value" in (p.get("text") or "").lower() or "held by non-affiliates" in (p.get("text") or "").lower() for p in prev_window)
            has_docs = any("documents incorporated" in (n.get("text") or "").lower() for n in next_window)
            if has_market or has_docs:
                out.append(span)
    except Exception:
        return []
    return out
