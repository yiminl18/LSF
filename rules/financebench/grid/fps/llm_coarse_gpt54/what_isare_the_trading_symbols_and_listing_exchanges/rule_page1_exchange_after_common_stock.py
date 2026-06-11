def rule_page1_exchange_after_common_stock(doc: dict) -> list[dict]:
    """Match exchange-name spans appearing shortly after a common-stock mention on page 1."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, s in enumerate(texts):
            if s.get("page_no") == 1 and re.search(r"Common Stock", (s.get("text") or "") + " " + (s.get("text_span") or ""), re.I):
                for j in range(i + 1, min(len(texts), i + 8)):
                    if texts[j].get("page_no") == 1 and re.search(r"NASDAQ|NYSE|New York Stock Exchange|Nasdaq", (texts[j].get("text") or ""), re.I):
                        out.append(texts[j])
        return out
    except Exception:
        return []
