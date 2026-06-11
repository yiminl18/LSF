def rule_table_with_year_end_columns_and_debt(doc: dict) -> list[dict]:
    """Match tables that have year columns and debt-related row labels, typical of year-end balance sheet disclosures."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = span.get("table_data", {}).get("cells", [])
            has_year = any(c.get("is_column_header") and re.search(r"\b20\d{2}\b", c.get("text") or "") for c in cells)
            has_debt_row = any(c.get("is_row_header") and re.search(r"\bdebt\b|\bborrowings\b", c.get("text") or "", re.I) for c in cells)
            if has_year and has_debt_row:
                out.append(span)
        return out
    except Exception:
        return []
