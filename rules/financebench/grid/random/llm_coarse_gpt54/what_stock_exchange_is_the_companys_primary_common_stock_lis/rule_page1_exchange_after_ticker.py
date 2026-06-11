def rule_page1_exchange_after_ticker(doc: dict) -> list[dict]:
    """Return spans immediately after likely ticker spans on page 1, where the exchange often follows."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        ticker_re = re.compile(r"^[A-Z]{2,6}([./-][A-Z0-9]+)?$")
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and ticker_re.match(txt):
                for j in range(i + 1, min(i + 4, len(texts))):
                    s = texts[j]
                    if s.get("page_no") != 1:
                        break
                    low = (s.get("text") or "").lower()
                    if "stock exchange" in low or "nasdaq" in low:
                        out.append(s)
        return out
    except Exception:
        return []
