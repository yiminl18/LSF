def rule_tables_with_total_assets_and_total_current_liabilities(doc: dict) -> list[dict]:
    """Match balance-sheet-like tables containing total assets and total current liabilities."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            low = (span.get("text") or "").lower()
            if "total assets" in low and ("total current liabilities" in low or "current liabilities" in low):
                out.append(span)
        return out
    except Exception:
        return []
