def rule_table_total_assets_balance_sheet(doc: dict) -> list[dict]:
    """Match first 2 tables with total assets row header in balance sheet/financial sections."""
    try:
        results = []
        path_keywords = ["item 6", "item 8", "balance sheet", "selected financial",
                        "financial statement", "consolidated balance", "annual report", "part iv"]
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path_text = span.get("structure", {}).get("path_text", "").lower()
            if not any(kw in path_text for kw in path_keywords):
                continue
            cells = span.get("table_data", {}).get("cells", [])
            row_headers = [c.get("text", "").lower() for c in cells if c.get("is_row_header")]
            for h in row_headers:
                if h.strip() in ["total assets", "total assets (i)"]:
                    results.append(span)
                    break
            if len(results) >= 2:
                break
        return results
    except Exception:
        return []
