def rule_table_revenue_financial_sections(doc: dict) -> list[dict]:
    """Match first 2 tables with revenue/sales row headers in financial sections (Item 6, 7, 8)."""
    try:
        results = []
        revenue_keywords = ["revenue", "net sales", "total sales", "sales to customer"]
        path_keywords = ["item 6", "item 7", "item 8", "selected financial",
                        "management's discussion", "financial statement",
                        "consolidated statement", "statement of income",
                        "statement of earnings", "statement of operations"]
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path_text = span.get("structure", {}).get("path_text", "").lower()
            if not any(kw in path_text for kw in path_keywords):
                continue
            cells = span.get("table_data", {}).get("cells", [])
            row_headers = [c.get("text", "").lower() for c in cells if c.get("is_row_header")]
            for h in row_headers:
                if any(kw in h for kw in revenue_keywords):
                    results.append(span)
                    break
            if len(results) >= 2:
                break
        return results
    except Exception:
        return []
