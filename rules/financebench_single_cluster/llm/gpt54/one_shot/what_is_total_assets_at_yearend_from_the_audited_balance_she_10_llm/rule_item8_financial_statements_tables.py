def rule_item8_financial_statements_tables(doc: dict) -> list[dict]:
    """Match tables under Item 8 / Financial Statements and Supplementary Data, where audited balance sheets usually appear."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "item 8" in path or "financial statements and supplementary data" in path:
                out.append(span)
        return out
    except Exception:
        return []
