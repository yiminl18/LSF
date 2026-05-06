def rule_table_long_term_debt_header(doc: dict) -> list[dict]:
    """Match tables with row headers containing 'long-term debt' or 'long-term obligations'."""
    try:
        results = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = span.get("table_data", {}).get("cells", [])
            row_headers = [c.get("text", "").lower() for c in cells if c.get("is_row_header")]
            for h in row_headers:
                if "long-term debt" in h or "long term debt" in h or "long-term obligations" in h:
                    results.append(span)
                    break
        return results
    except Exception:
        return []
