def rule_page1_company_common_stock_exchange_sequence(doc: dict) -> list[dict]:
    """Match page-1 sequences containing company cover info, common stock, symbol, and exchange."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts)):
            if texts[i].get("page_no") != 1:
                continue
            window = texts[i:min(len(texts), i + 20)]
            joined = " ".join((w.get("text") or "") + " " + (w.get("text_span") or "") for w in window)
            if re.search(r"Common Stock", joined, re.I) and re.search(r"NASDAQ|NYSE|New York Stock Exchange|Nasdaq", joined, re.I):
                if re.search(r"\b[A-Z]{1,6}(?:\d+[A-Z]*)?(?:[./-][A-Z0-9]+)?\b", joined):
                    out.extend(window)
        return out
    except Exception:
        return []
