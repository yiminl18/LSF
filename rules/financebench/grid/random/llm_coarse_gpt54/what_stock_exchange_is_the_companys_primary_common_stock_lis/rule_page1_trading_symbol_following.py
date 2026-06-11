def rule_page1_trading_symbol_following(doc: dict) -> list[dict]:
    """Return spans near 'Trading Symbol' / 'Trading Symbol(s)' on page 1, where exchange often appears adjacent."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and ("trading symbol" in txt or "trading symbol(s)" in txt):
                for j in range(i, min(i + 8, len(texts))):
                    s = texts[j]
                    if s.get("page_no") != 1:
                        break
                    out.append(s)
        return out
    except Exception:
        return []
