def rule_after_trading_symbol_on_page1(doc: dict) -> list[dict]:
    """Match exchange spans appearing shortly after a 'Trading Symbol' anchor on page 1."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        anchors = []
        for i, s in enumerate(texts):
            combined = " ".join([s.get("text", "") or "", s.get("text_span", "") or ""])
            if s.get("page_no") == 1 and re.search(r'trading symbol', combined, re.I):
                anchors.append(i)
        for idx in anchors:
            for j in range(idx, min(len(texts), idx + 8)):
                s = texts[j]
                if s.get("page_no") != 1:
                    continue
                t = (s.get("text") or "").strip()
                if re.search(r'new york stock exchange|nasdaq|global select market', t, re.I):
                    out.append(s)
        return out
    except Exception:
        return []
