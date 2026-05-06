def rule_tables_with_total_assets_and_current_assets(doc: dict) -> list[dict]:
    """Match tables containing both current assets and total assets, a strong balance-sheet signature."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "").lower()
            if "total assets" in text and "current assets" in text:
                out.append(span)
        return out
    except Exception:
        return []
