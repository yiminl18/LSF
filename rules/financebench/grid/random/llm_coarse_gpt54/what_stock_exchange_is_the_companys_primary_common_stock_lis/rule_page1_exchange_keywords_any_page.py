def rule_page1_exchange_keywords_any_page(doc: dict) -> list[dict]:
    """Match spans anywhere in the document whose text looks like a stock exchange name."""
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
            txt = (span.get("text") or "")
            low = txt.lower()
            if any(re.search(p, low) for p in pats):
                out.append(span)
        return out
    except Exception:
        return []
