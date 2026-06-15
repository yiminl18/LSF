def rule_common_stock_listing_sentence(doc: dict) -> list[dict]:
    """Match narrative sentences stating where the common stock is traded, listed, or principally marketed."""
    try:
        patterns = ("traded on", "listed on", "principal market")
        return [
            s for s in doc.get("texts", [])
            if "common stock" in s.get("text", "").lower()
            and any(p in s.get("text", "").lower() for p in patterns)
        ]
    except Exception:
        return []
