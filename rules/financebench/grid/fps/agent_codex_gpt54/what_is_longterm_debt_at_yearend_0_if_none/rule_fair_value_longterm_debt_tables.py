def rule_fair_value_longterm_debt_tables(doc: dict) -> list[dict]:
    """Match fair-value tables that explicitly include long-term debt balances."""
    try:
        results = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue

            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            text = (span.get("text") or "").lower()
            if (
                any(marker in path for marker in ("fair value of financial instruments", "fair value measurements"))
                and "long-term debt" in text
            ):
                results.append(span)

        return results
    except Exception:
        return []
