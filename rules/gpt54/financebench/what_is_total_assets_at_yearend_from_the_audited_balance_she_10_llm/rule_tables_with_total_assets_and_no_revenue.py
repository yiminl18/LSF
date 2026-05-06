def rule_tables_with_total_assets_and_no_revenue(doc: dict) -> list[dict]:
    """Match total-assets tables while excluding obvious revenue/sales tables."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            low = (span.get("text") or "").lower()
            if "total assets" in low and "net sales" not in low and "revenue" not in low and "revenues" not in low:
                out.append(span)
        return out
    except Exception:
        return []
