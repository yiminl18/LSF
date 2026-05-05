def rule_tables_with_assets_and_current_liabilities(doc: dict) -> list[dict]:
    """Match balance-sheet-like tables containing assets and current liabilities."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            low = (span.get("text") or "").lower()
            if "assets" in low and "current liabilities" in low:
                out.append(span)
        return out
    except Exception:
        return []
