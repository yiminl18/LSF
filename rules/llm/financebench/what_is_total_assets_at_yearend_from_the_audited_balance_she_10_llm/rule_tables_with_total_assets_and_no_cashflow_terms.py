def rule_tables_with_total_assets_and_no_cashflow_terms(doc: dict) -> list[dict]:
    """Match total-assets tables while excluding obvious cash flow statements."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if "total assets" in txt and "cash flows" not in txt and "operating activities" not in txt:
                out.append(span)
        return out
    except Exception:
        return []
