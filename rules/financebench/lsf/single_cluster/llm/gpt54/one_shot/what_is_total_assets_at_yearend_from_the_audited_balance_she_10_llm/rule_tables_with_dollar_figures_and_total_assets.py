def rule_tables_with_dollar_figures_and_total_assets(doc: dict) -> list[dict]:
    """Match tables containing total assets and dollar-sign figures."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text") or ""
            if "total assets" in text.lower() and "$" in text:
                out.append(span)
        return out
    except Exception:
        return []
