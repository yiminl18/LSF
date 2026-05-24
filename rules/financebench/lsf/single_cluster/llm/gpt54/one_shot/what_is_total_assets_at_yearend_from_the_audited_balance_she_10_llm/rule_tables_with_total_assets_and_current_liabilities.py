def rule_tables_with_total_assets_and_current_liabilities(doc: dict) -> list[dict]:
    """Match balance-sheet-like tables containing total assets and current liabilities."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "").lower()
            if "total assets" in text and "current liabilities" in text:
                out.append(span)
        return out
    except Exception:
        return []
