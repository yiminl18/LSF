def rule_page1_2_section12b_exchange_table(doc: dict) -> list[dict]:
    """Match page 1-2 Section 12(b) tables with trading-symbol and exchange columns."""
    try:
        return [
            s for s in doc.get("texts", [])
            if s.get("page_no", 999) <= 2
            and s.get("label") == "table"
            and "trading symbol" in s.get("text", "").lower()
            and "exchange" in s.get("text", "").lower()
        ]
    except Exception:
        return []
