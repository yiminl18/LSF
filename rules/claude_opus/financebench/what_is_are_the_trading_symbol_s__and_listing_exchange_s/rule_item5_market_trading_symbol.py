def rule_item5_market_trading_symbol(doc: dict) -> list[dict]:
    """Match spans in Item 5 (Market for Registrant's Common Stock) mentioning trading symbol."""
    try:
        import re
        results = []
        for span in doc.get("texts", []):
            path = span.get("structure", {}).get("path_text", "").lower()
            text = span.get("text", "").lower()

            if "item 5" in path or "market for" in path:
                if re.search(r'\b(symbol|traded|trading|nasdaq|nyse|stock exchange)\b', text):
                    results.append(span)
        return results
    except Exception:
        return []
