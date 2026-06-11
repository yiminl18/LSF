def rule_page1_common_stock_and_symbol(doc: dict) -> list[dict]:
    """Match page-1 spans mentioning common stock together with a nearby ticker-like symbol."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, s in enumerate(texts):
            txt = (s.get("text") or "") + " " + (s.get("text_span") or "")
            if s.get("page_no") == 1 and re.search(r"Common Stock", txt, re.I):
                window = texts[max(0, i - 2):min(len(texts), i + 8)]
                if any(re.fullmatch(r"[A-Z0-9./-]{1,15}", (w.get("text") or "").strip()) for w in window):
                    out.extend(window)
        return out
    except Exception:
        return []
