def rule_balance_sheet_debt_row_in_item_8(doc: dict) -> list[dict]:
    """Match tables under Item 8 / Financial Statements and Supplementary Data that contain long-term debt rows."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "item 8" not in path and "financial statements and supplementary data" not in path:
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            joined = " ".join((c.get("text") or "") for c in cells).lower()
            if re.search(r"\blong[\-\s]?term debt\b", joined) or "debt excluding current maturities" in joined:
                out.append(span)
        return out
    except Exception:
        return []
