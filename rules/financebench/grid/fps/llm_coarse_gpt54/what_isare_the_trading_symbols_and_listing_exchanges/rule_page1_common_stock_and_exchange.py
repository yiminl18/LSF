def rule_page1_common_stock_and_exchange(doc: dict) -> list[dict]:
    """Match page-1 spans mentioning common stock together with a nearby exchange name."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, s in enumerate(texts):
            txt = (s.get("text") or "") + " " + (s.get("text_span") or "")
            if s.get("page_no") == 1 and re.search(r"Common Stock", txt, re.I):
                window = texts[max(0, i - 2):min(len(texts), i + 8)]
                if any(re.search(r"NASDAQ|NYSE|New York Stock Exchange|Nasdaq", (w.get("text") or ""), re.I) for w in window):
                    out.extend(window)
        return out
    except Exception:
        return []
