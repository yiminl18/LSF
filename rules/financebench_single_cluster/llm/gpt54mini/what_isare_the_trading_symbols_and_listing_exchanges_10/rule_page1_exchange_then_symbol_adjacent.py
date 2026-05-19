def rule_page1_exchange_then_symbol_adjacent(doc: dict) -> list[dict]:
    """Match adjacent page 1 spans where an exchange name is near a ticker-like token."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts) - 1):
            a, b = texts[i], texts[i + 1]
            if a.get("page_no") == 1 and b.get("page_no") == 1:
                at = ((a.get("text") or "") + " " + (a.get("text_span") or "")).lower()
                bt = (b.get("text") or "").strip()
                if ("nasdaq" in at or "new york stock exchange" in at or "nyse" in at or "exchange" in at) and re.fullmatch(r"[A-Z]{1,6}[0-9A-Z]{0,4}", bt):
                    out.extend([a, b])
        return out
    except Exception:
        return []
