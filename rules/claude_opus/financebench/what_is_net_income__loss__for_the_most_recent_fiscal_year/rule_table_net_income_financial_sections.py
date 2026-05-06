def rule_table_net_income_financial_sections(doc: dict) -> list[dict]:
    """Match first 3 tables with net income/earnings headers in financial sections (Item 6, 7, 8)."""
    try:
        results = []
        financial_keywords = ["item 6", "item 7", "item 8", "selected financial",
                            "financial statement", "management's discussion"]
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path_text = span.get("structure", {}).get("path_text", "").lower()
            if not any(kw in path_text for kw in financial_keywords):
                continue
            cells = span.get("table_data", {}).get("cells", [])
            row_headers = [c.get("text", "").lower() for c in cells if c.get("is_row_header")]
            for h in row_headers:
                if ("net income" in h or "net earnings" in h or "net loss" in h) and "per share" not in h:
                    results.append(span)
                    break
            if len(results) >= 3:
                break
        return results
    except Exception:
        return []
