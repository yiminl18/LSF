def rule_page1_2_short_exchange_values(doc: dict) -> list[dict]:
    """Match short page-1/2 exchange-value spans and release-style bracketed exchange tickers."""
    try:
        out = []
        for s in doc.get("texts", []):
            if s.get("page_no", 999) > 2:
                continue
            text = s.get("text", "")
            low = text.lower().strip()
            if "[nyse:" in low or "[nasdaq:" in low:
                out.append(s)
                continue
            if (
                ("nasdaq" in low or "new york stock exchange" in low)
                and len(low.split()) <= 20
            ):
                out.append(s)
        return out
    except Exception:
        return []
