def rule_page1_symbol_then_exchange_adjacent(doc: dict) -> list[dict]:
    """Match adjacent page 1 spans where a ticker-like token is followed by an exchange name."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts) - 1):
            a, b = texts[i], texts[i + 1]
            if a.get("page_no") == 1 and b.get("page_no") == 1:
                at = (a.get("text") or "").strip()
                bt = ((b.get("text") or "") + " " + (b.get("text_span") or "")).lower()
                if re.fullmatch(r"[A-Z]{1,6}[0-9A-Z]{0,4}", at) and (
                    "nasdaq" in bt or "new york stock exchange" in bt or "nyse" in bt or "exchange" in bt
                ):
                    out.extend([a, b])
        return out
    except Exception:
        return []
