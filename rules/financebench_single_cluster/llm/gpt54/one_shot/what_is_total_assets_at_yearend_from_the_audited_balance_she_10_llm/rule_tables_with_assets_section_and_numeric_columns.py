def rule_tables_with_assets_section_and_numeric_columns(doc: dict) -> list[dict]:
    """Match tables with assets language and multiple numeric-looking cells."""
    import re
    try:
        out = []
        num_re = re.compile(r"\d")
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "").lower()
            if "assets" not in text:
                continue
            numeric_cells = 0
            for c in (span.get("table_data") or {}).get("cells", []):
                if num_re.search(c.get("text") or ""):
                    numeric_cells += 1
            if numeric_cells >= 4:
                out.append(span)
        return out
    except Exception:
        return []
