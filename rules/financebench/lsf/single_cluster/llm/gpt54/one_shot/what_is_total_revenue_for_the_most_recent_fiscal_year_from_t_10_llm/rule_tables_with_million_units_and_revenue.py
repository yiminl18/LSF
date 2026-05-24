def rule_tables_with_million_units_and_revenue(doc: dict) -> list[dict]:
    """Match financial tables mentioning millions/billions and revenue-like rows."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if any(u in txt for u in ["million", "millions", "billion", "billions"]) and any(
                k in txt for k in ["revenue", "revenues", "sales", "net sales", "net revenues"]
            ):
                out.append(span)
        return out
    except Exception:
        return []
