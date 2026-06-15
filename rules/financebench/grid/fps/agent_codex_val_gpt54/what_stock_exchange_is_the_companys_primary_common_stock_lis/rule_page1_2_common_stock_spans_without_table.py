def rule_page1_2_common_stock_spans_without_table(doc: dict) -> list[dict]:
    """Match common-stock title spans on page 1-2 when no clean Section 12(b) table is present."""
    try:
        texts = doc.get("texts", [])
        has_table = any(
            s.get("page_no", 999) <= 2
            and s.get("label") == "table"
            and "trading symbol" in s.get("text", "").lower()
            and "exchange" in s.get("text", "").lower()
            for s in texts
        )
        if has_table:
            return []

        keys = ("common stock", "ordinary shares", "class a common stock")
        return [
            s for s in texts
            if s.get("page_no", 999) <= 2
            and any(k in s.get("text", "").lower() for k in keys)
        ]
    except Exception:
        return []
