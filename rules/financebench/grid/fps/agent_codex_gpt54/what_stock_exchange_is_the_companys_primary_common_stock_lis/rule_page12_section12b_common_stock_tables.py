def rule_page12_section12b_common_stock_tables(doc: dict) -> list[dict]:
    """Match page-1/2 Section 12(b) tables that include the common-stock listing row."""
    try:
        return [
            s
            for s in doc.get("texts", [])
            if s.get("page_no", 999) <= 2
            and s.get("label") == "table"
            and "exchange on which registered" in s.get("text", "").lower()
            and (
                "common stock" in s.get("text", "").lower()
                or "ordinary shares" in s.get("text", "").lower()
            )
        ]
    except Exception:
        return []
