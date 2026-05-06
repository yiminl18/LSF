def rule_page1_topmatter_with_common_stock_and_exchange(doc: dict) -> list[dict]:
    """Match page 1 spans mentioning common stock together with exchange/registration cues."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r"common stock", txt, re.I) and re.search(r"exchange|registered|symbol|nasdaq|new york stock exchange|nyse", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
