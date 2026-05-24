def rule_item8_financial_statements_income_statement(doc: dict) -> list[dict]:
    """Retrieve spans in Item 8 / financial statements areas, especially income statement and revenue tables."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            label = span.get("label") or ""
            hay = (path + "\n" + text).lower()
            item8 = (
                "item 8" in hay
                or "financial statements and supplementary data" in hay
                or "financial statements" in hay
                or "supplementary data" in hay
            )
            income_stmt = (
                "statement of income" in hay
                or "statements of income" in hay
                or "income statement" in hay
                or "statements of earnings" in hay
                or "statement of earnings" in hay
                or "statement of operations" in hay
                or "statements of operations" in hay
                or "consolidated revenue" in hay
                or "net sales" in hay
                or "total revenue" in hay
                or "sales to customers" in hay
            )
            if (item8 and (label in {"table", "text", "section_header"})) or income_stmt:
                out.append(span)
        return out
    except Exception:
        return []
