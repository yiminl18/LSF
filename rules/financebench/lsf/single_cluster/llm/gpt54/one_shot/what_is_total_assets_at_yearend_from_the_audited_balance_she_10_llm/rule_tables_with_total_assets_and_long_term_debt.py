def rule_tables_with_total_assets_and_long_term_debt(doc: dict) -> list[dict]:
    """Match balance-sheet-like tables containing total assets and long-term debt/borrowings."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            low = (span.get("text") or "").lower()
            if "total assets" in low and ("long-term debt" in low or "long term debt" in low or "borrowings" in low):
                out.append(span)
        return out
    except Exception:
        return []
