def rule_balance_sheet_assets_liabilities_table(doc: dict) -> list[dict]:
    """Retrieve likely balance-sheet tables by assets/liabilities row cues in financial statement sections."""
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
                or "financial statements" in path
                or "supplementary data" in path
                or "selected financial data" in path
            )
            has_assets = "assets" in cell_text or "assets" in text
            has_liab = "liabilities" in cell_text or "liabilities" in text
            has_equity = "equity" in cell_text or "shareholders' equity" in cell_text or "stockholders' equity" in cell_text or "equity" in text
            if in_fin and has_assets and (has_liab or has_equity):
                out.append(span)
        return out
    except Exception:
        return []

