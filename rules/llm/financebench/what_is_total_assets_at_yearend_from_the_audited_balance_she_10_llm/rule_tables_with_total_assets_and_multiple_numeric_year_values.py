def rule_tables_with_total_assets_and_multiple_numeric_year_values(doc: dict) -> list[dict]:
    """Match total-assets tables with multiple numeric values, suggesting year-end amounts."""
    import re
    try:
        out = []
        num_re = re.compile(r"\d")
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            if "total assets" not in (span.get("text") or "").lower():
                continue
            count = 0
            for c in (span.get("table_data") or {}).get("cells", []):
                if num_re.search(c.get("text") or ""):
                    count += 1
            if count >= 6:
                out.append(span)
        return out
    except Exception:
        return []
