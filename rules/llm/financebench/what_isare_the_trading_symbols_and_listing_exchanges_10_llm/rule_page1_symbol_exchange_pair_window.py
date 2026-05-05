def rule_page1_symbol_exchange_pair_window(doc: dict) -> list[dict]:
    """Match windows on page 1 where a ticker-like token and an exchange name co-occur within a few spans."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts)):
            window = texts[i:i+6]
            if not window or any(s.get("page_no") != 1 for s in window):
                continue
            joined = " ".join(((s.get("text") or "") + " " + (s.get("text_span") or "")) for s in window)
            has_exchange = any(x in joined.lower() for x in ["nasdaq", "new york stock exchange", "nyse"])
            has_symbol = any(re.fullmatch(r"[A-Z]{1,6}[0-9A-Z]{0,4}", (s.get("text") or "").strip() or "") for s in window)
            if has_exchange and has_symbol:
                out.extend(window)
        return out
    except Exception:
        return []
