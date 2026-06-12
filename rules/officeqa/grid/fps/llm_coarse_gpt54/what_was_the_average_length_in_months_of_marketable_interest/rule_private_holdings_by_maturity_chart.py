def rule_private_holdings_by_maturity_chart(doc: dict) -> list[dict]:
    """Match chart/title spans about private holdings of Treasury marketable debt by maturity, often adjacent to the answer table."""
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            if "private holdings" in txt and "marketable debt" in txt and "maturity" in txt:
                out.append(span)
    except Exception:
        return []
    return out
