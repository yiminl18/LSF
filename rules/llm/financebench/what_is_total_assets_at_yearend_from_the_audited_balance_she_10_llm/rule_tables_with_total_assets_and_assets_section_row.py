def rule_tables_with_total_assets_and_assets_section_row(doc: dict) -> list[dict]:
    """Match tables where total assets appears after several asset-related rows."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (span.get("table_data") or {}).get("cells") or []
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c)
            asset_rows_before = 0
            found = False
            for r in sorted(rows):
                row_text = " ".join((c.get("text") or "").lower() for c in rows[r])
                if "total assets" in row_text:
                    found = True
                    break
                if "assets" in row_text or "cash" in row_text or "inventory" in row_text or "receivable" in row_text:
                    asset_rows_before += 1
            if found and asset_rows_before >= 2:
                out.append(span)
        return out
    except Exception:
        return []
