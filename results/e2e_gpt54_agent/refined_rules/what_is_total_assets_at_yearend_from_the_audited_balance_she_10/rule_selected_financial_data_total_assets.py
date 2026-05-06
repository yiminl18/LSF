def rule_selected_financial_data_total_assets(doc: dict) -> list[dict]:
    """Retrieve Item 6 selected financial data tables containing total assets."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            text = (span.get("text") or "").lower()
            if "item 6" not in path and "selected financial data" not in path:
                continue
            if "selected financial data" not in path and "selected financial data" not in text:
                continue
            if "total assets" in text or "assets" in text:
                out.append(span)
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            cell_text = " ".join((c.get("text") or "") for c in cells).lower()
            if "total assets" in cell_text:
                out.append(span)
        return out
    except Exception:
        return []

