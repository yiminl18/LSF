def rule_balance_sheet_longterm_debt_tables(doc: dict) -> list[dict]:
    """Match balance-sheet tables that include a long-term debt or term debt row."""
    try:
        results = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue

            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            text = (span.get("text") or "").lower()
            if (
                ("balance sheet" in path or "statements of financial position" in path)
                and any(marker in text for marker in ("long-term debt", "long term debt", "term debt", "debt, net"))
            ):
                results.append(span)

        return results
    except Exception:
        return []
