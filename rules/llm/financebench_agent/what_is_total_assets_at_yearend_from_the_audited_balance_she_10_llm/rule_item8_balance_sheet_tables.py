def rule_item8_balance_sheet_tables(doc: dict) -> list[dict]:
    """Retrieve balance-sheet tables under Item 8 / financial statements sections."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            t = text.lower()
            p = path.lower()
            item8 = (
                "item 8" in p
                or "financial statements and supplementary data" in p
                or "financial statements" in p
                or "supplementary data" in p
            )
            balance = (
                "balance sheet" in t
                or "balance sheets" in t
                or "statement of financial position" in t
                or "statement of financial positions" in t
            )
            assets = "total assets" in t or "assets" in t
            if item8 and balance and assets:
                out.append(span)
        return out
    except Exception:
        return []

