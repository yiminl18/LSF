def rule_tables_with_exchange_rate_changes_for_dollar(doc: dict) -> list[dict]:
    """Match tables for exchange-rate statistics that often neighbor the answer-bearing section."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "").lower()
                if "exchange rate changes for the dollar" in txt or "trade-weighted index of foreign currency value of the dollar" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
