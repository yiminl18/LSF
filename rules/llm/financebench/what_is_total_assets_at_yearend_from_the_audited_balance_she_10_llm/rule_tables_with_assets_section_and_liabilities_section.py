def rule_tables_with_assets_section_and_liabilities_section(doc: dict) -> list[dict]:
    """Match tables with explicit assets and liabilities sections in separate rows."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            row_texts = {}
            for c in cells:
                row_texts.setdefault(c.get("row"), "")
                row_texts[c.get("row")] += " " + (c.get("text") or "").lower()
            has_assets_row = any(rt.strip() == "assets" or " assets " in f" {rt} " for rt in row_texts.values())
            has_liab_row = any("liabilities" in rt for rt in row_texts.values())
            if has_assets_row and has_liab_row:
                out.append(span)
        return out
    except Exception:
        return []
