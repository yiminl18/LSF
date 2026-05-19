def rule_page1_title_trading_exchange_cluster(doc: dict) -> list[dict]:
    """Match spans in the page 1 cluster containing title/trading symbol/exchange labels and nearby values."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "")
            if span.get("page_no") == 1 and re.search(r"title of each class|name of each exchange|trading symbol", txt, re.I):
                for j in range(max(0, i - 2), min(len(texts), i + 8)):
                    s = texts[j]
                    t = (s.get("text") or "")
                    if re.search(r"title of each class|name of each exchange|trading symbol|nasdaq|new york stock exchange|nyse", t, re.I) or re.fullmatch(r"[A-Z]{1,8}\d*[A-Z]*", t.strip()):
                        out.append(s)
        return out
    except Exception:
        return []
