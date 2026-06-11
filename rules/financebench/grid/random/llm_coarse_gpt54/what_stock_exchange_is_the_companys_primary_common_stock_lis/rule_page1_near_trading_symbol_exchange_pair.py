def rule_page1_near_trading_symbol_exchange_pair(doc: dict) -> list[dict]:
    """Return spans in page-1 local windows where a trading symbol and exchange name co-occur."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        exchange_re = re.compile(r"\b(new york stock exchange|the new york stock exchange|nasdaq|nasdaq global select market|the nasdaq global select market)\b")
        for i in range(len(texts)):
            if texts[i].get("page_no") != 1:
                continue
            window = texts[i:i+6]
            joined = " ".join((w.get("text") or "") for w in window).lower()
            if ("trading symbol" in joined or "trading symbol(s)" in joined) and exchange_re.search(joined):
                out.extend(window)
        return out
    except Exception:
        return []
