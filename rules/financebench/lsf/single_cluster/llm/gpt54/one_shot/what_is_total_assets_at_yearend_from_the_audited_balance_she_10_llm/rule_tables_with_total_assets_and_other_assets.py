def rule_tables_with_total_assets_and_other_assets(doc: dict) -> list[dict]:
    """Match balance-sheet-like tables containing total assets and other assets."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            low = (span.get("text") or "").lower()
            if "total assets" in low and "other assets" in low:
                out.append(span)
        return out
    except Exception:
        return []
