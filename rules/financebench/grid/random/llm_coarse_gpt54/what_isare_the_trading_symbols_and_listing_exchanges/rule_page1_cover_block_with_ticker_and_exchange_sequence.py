def rule_page1_cover_block_with_ticker_and_exchange_sequence(doc: dict) -> list[dict]:
    """Return page-1 spans in a local sequence where ticker-like and exchange-like spans co-occur within a few siblings."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts)):
            if texts[i].get("page_no") != 1:
                continue
            window = texts[i:i+6]
            blob = " ".join((s.get("text", "") or "") for s in window)
            if re.search(r"\b[A-Z]{1,6}(?:[/-][A-Z0-9]{1,6})?\d{0,2}\b", blob) and re.search(r"nasdaq|stock exchange|nyse", blob, re.I):
                out.extend([s for s in window if s.get("page_no") == 1])
        return out
    except Exception:
        return []
