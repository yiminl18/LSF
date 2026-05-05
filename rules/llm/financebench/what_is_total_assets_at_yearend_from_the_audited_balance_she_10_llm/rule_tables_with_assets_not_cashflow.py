def rule_tables_with_assets_not_cashflow(doc: dict) -> list[dict]:
    """Match asset tables while excluding obvious cash flow / income statement tables."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "").lower()
            if "assets" not in text:
                continue
            if "cash flows" in text or "statement of income" in text or "comprehensive income" in text:
                continue
            out.append(span)
        return out
    except Exception:
        return []
