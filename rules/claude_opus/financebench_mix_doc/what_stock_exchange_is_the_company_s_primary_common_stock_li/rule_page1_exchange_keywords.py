def rule_page1_exchange_keywords(doc: dict) -> list[dict]:
    """Match page 1 spans containing stock exchange keywords (nasdaq, new york stock exchange)."""
    try:
        keywords = ["nasdaq", "new york stock exchange", "stock exchange"]
        return [
            s for s in doc.get("texts", [])
            if s.get("page_no") == 1 and
            any(kw in s.get("text", "").lower() for kw in keywords)
        ]
    except Exception:
        return []
