def rule_page1_exchange_keywords(doc: dict) -> list[dict]:
    """Match page-1 spans whose text looks like a stock exchange name."""
    try:
        import re
        texts = doc.get("texts", [])
        pats = [
            r"\bnew york stock exchange\b",
            r"\bthe new york stock exchange\b",
            r"\bnasdaq\b",
            r"\bnasdaq global select market\b",
            r"\bthe nasdaq global select market\b",
            r"\bnasdaq global market\b",
            r"\bthe nasdaq global market\b",
            r"\bnasdaq stock market\b",
            r"\bthe nasdaq stock market\b",
        ]
        out = []
        for span in texts:
            if span.get("page_no") == 1:
                txt = (span.get("text") or "")
                low = txt.lower()
                if any(re.search(p, low) for p in pats):
                    out.append(span)
        return out
    except Exception:
        return []
