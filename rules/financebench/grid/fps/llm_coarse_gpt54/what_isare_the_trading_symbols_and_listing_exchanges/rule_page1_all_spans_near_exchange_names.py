def rule_page1_all_spans_near_exchange_names(doc: dict) -> list[dict]:
    """Match a small neighborhood around page-1 exchange-name spans."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, s in enumerate(texts):
            if s.get("page_no") == 1 and re.search(r"NASDAQ|NYSE|New York Stock Exchange|Nasdaq", (s.get("text") or ""), re.I):
                for j in range(max(0, i - 3), min(len(texts), i + 4)):
                    if texts[j].get("page_no") == 1:
                        out.append(texts[j])
        return out
    except Exception:
        return []
