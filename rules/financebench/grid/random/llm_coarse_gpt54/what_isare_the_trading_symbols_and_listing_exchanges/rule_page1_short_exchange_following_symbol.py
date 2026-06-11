def rule_page1_short_exchange_following_symbol(doc: dict) -> list[dict]:
    """Match page-1 spans where a short ticker-like span is followed by an exchange-name span."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts) - 1):
            a = texts[i]
            b = texts[i + 1]
            ta = (a.get("text", "") or "").strip()
            tb = (b.get("text", "") or "").strip()
            if a.get("page_no") == 1 and b.get("page_no") == 1:
                if re.fullmatch(r"[A-Z]{1,6}(?:[/-][A-Z0-9]{1,6})?\d{0,2}", ta) and re.search(r"nasdaq|stock exchange|nyse", tb, re.I):
                    out.extend([a, b])
        return out
    except Exception:
        return []
