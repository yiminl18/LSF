def rule_page1_exchange_near_ordinary_shares(doc: dict) -> list[dict]:
    """Match page-1 spans in local windows where 'ordinary shares' and an exchange name co-occur."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        ex_re = re.compile(r"\b(new york stock exchange|the new york stock exchange|nasdaq|nasdaq global select market|the nasdaq global select market)\b")
        for i in range(len(texts)):
            if texts[i].get("page_no") != 1:
                continue
            window = texts[i:i+8]
            joined = " ".join((w.get("text") or "") for w in window).lower()
            if "ordinary shares" in joined and ex_re.search(joined):
                out.extend(window)
        return out
    except Exception:
        return []
