def rule_tables_with_total_assets_and_no_income_statement_terms(doc: dict) -> list[dict]:
    """Match total-assets tables while excluding obvious income statements."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if "total assets" in txt and "net sales" not in txt and "net income" not in txt and "revenue" not in txt:
                out.append(span)
        return out
    except Exception:
        return []
