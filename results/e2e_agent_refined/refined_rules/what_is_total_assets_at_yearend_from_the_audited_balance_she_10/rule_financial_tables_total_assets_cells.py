def rule_financial_tables_total_assets_cells(doc: dict) -> list[dict]:
    """Retrieve financial statement tables whose cells include total assets under Item 8 or similar financial sections."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            text = (span.get("text") or "").lower()
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            cell_text = " ".join((c.get("text") or "") for c in cells).lower()
            in_fin = (
                "item 8" in path
                or "financial statements and supplementary data" in path
                or "financial statements" in path
                or "supplementary data" in path
                or "consolidated balance sheet" in path
                or "consolidated balance sheets" in path
            )
            has_assets = "total assets" in cell_text or "total assets" in text
            has_statement_cue = (
                "balance sheet" in cell_text
                or "balance sheets" in cell_text
                or "statement of financial position" in cell_text
                or "statement of financial positions" in cell_text
                or "assets" in cell_text
            )
            if in_fin and has_assets and has_statement_cue:
                out.append(span)
        return out
    except Exception:
        return []

