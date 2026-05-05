def rule_tables_with_total_assets_and_total_current_assets(doc: dict) -> list[dict]:
    """Match tables containing both total current assets/current assets and total assets."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            low = (span.get("text") or "").lower()
            if "total assets" in low and ("current assets" in low or "total current assets" in low):
                out.append(span)
        return out
    except Exception:
        return []
