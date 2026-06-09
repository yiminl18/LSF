def rule_selected_financial_data_longterm_debt_or_obligations(doc: dict) -> list[dict]:
    """Match Item 6 or selected-financial-data tables that summarize long-term debt or obligations."""
    try:
        hits: list[dict] = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = " ".join((span.get("text") or "").split()).lower()
            path = " ".join((((span.get("structure") or {}).get("path_text")) or "").split()).lower()
            if "selected financial data" not in path and "selected consolidated financial data" not in path:
                continue
            if (
                "long-term obligations" in text
                or "long-term debt (including capital lease obligations)" in text
                or "debt, non-current" in text
                or "long-term debt, excluding current portion" in text
            ):
                hits.append(span)
        return hits
    except Exception:
        return []
