def rule_page1_exchange_followed_by_trading_symbol_header(doc: dict) -> list[dict]:
    """Match page-1 spans in malformed OCR where exchange and trading-symbol labels may be swapped or adjacent."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts) - 2):
            window = texts[i:i+3]
            if all(s.get("page_no") == 1 for s in window):
                blob = " ".join((s.get("text", "") or "") for s in window)
                if re.search(r"trading symbol", blob, re.I) and re.search(r"stock exchange|nasdaq|nyse", blob, re.I):
                    out.extend(window)
        return out
    except Exception:
        return []
