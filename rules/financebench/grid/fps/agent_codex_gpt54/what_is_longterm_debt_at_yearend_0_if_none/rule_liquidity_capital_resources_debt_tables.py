def rule_liquidity_capital_resources_debt_tables(doc: dict) -> list[dict]:
    """Match liquidity or capital-resources summary tables that report long-term debt."""
    try:
        results = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue

            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            text = (span.get("text") or "").lower()
            if (
                any(marker in path for marker in ("liquidity and capital resources", "capital resources", "key balance sheet data"))
                and any(marker in text for marker in ("long-term debt", "term debt", "total debt"))
            ):
                results.append(span)

        return results
    except Exception:
        return []
