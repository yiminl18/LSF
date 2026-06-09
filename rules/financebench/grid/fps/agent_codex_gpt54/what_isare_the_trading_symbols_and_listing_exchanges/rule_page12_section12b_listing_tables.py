def rule_page12_section12b_listing_tables(doc: dict) -> list[dict]:
    """Match page-1/2 Section 12(b) tables containing trading-symbol and exchange headers."""
    try:
        return [
            s
            for s in doc.get("texts", [])
            if s.get("page_no", 999) <= 2
            and s.get("label") == "table"
            and "trading symbol" in s.get("text", "").lower()
            and (
                "name of each exchange on which registered" in s.get("text", "").lower()
                or "name of exchange on which registered" in s.get("text", "").lower()
                or "exchange on which registered" in s.get("text", "").lower()
            )
        ]
    except Exception:
        return []
