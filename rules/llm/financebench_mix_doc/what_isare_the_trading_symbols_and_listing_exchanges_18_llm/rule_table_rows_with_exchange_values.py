def rule_table_rows_with_exchange_values(doc: dict) -> list[dict]:
    """Match table spans containing rows with exchange values like NYSE/NASDAQ."""
    try:
        texts = doc.get("texts", [])
        out = []
        exchange_terms = [
            "new york stock exchange",
            "nasdaq",
            "chicago stock exchange",
        ]
        for span in texts:
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            if any((c.get("text") or "").lower() in exchange_terms for c in cells):
                out.append(span)
        return out
    except Exception:
        return []
