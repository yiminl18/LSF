def rule_page1_text_spans_with_common_stock_and_exchange(doc: dict) -> list[dict]:
    """Match page-1 spans mentioning common stock together with an exchange name."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = ((span.get("text", "") or "") + " " + (span.get("text_span", "") or "")).strip()
            if span.get("page_no") == 1 and re.search(r"common stock", txt, re.I) and re.search(r"nasdaq|stock exchange|nyse", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
