def rule_selected_financial_data_debt_rows(doc: dict) -> list[dict]:
    """Match selected-financial-data tables that expose a non-current debt row."""
    try:
        results = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue

            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            text = (span.get("text") or "").lower()
            if "selected financial data" not in path:
                continue
            if any(
                marker in text
                for marker in ("non-current portion of term debt", "long-term debt, net and other long-term liabilities")
            ):
                results.append(span)

        return results
    except Exception:
        return []
