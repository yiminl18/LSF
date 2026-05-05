def rule_item8_income_statement_tables(doc: dict) -> list[dict]:
    """Match tables under Item 8 / Financial Statements that look like audited income statements."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if "Item 8" not in path and "Financial Statements" not in path and "Supplementary Data" not in path:
                continue
            hay = (path + "\n" + text).lower()
            if (
                "statement of income" in hay
                or "statements of income" in hay
                or "statement of earnings" in hay
                or "statements of earnings" in hay
                or "statement of operations" in hay
                or "statements of operations" in hay
                or "income statement" in hay
            ):
                out.append(span)
        return out
    except Exception:
        return []
