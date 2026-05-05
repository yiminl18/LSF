def rule_page1_exchange_header_following_symbol_header(doc: dict) -> list[dict]:
    """Match exchange header spans that appear shortly after a trading symbol header on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "trading symbol" in txt:
                for j in range(i + 1, min(len(texts), i + 6)):
                    s = texts[j]
                    stxt = (s.get("text") or "").lower()
                    if s.get("page_no") == 1 and (
                        "exchange on which registered" in stxt
                        or "name of each exchange" in stxt
                    ):
                        out.append(s)
        return out
    except Exception:
        return []
