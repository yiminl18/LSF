def rule_page1_standalone_number_after_market_value_and_outstanding_label(doc: dict) -> list[dict]:
    """Match page-1 standalone numeric spans in two-column cover-page layouts where market value and outstanding count are listed separately."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        num_re = re.compile(r"^\d[\d,]{5,}$")
        for i, span in enumerate(texts):
            text = (span.get("text") or "").strip()
            if span.get("page_no") != 1 or not num_re.match(text):
                continue
            prev = " ".join((p.get("text") or "") for p in texts[max(0, i - 8):i]).lower()
            if "number of shares of common stock outstanding" in prev or "shares of common stock outstanding as of" in prev:
                out.append(span)
    except Exception:
        return []
    return out
