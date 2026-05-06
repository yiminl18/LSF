def rule_tables_with_total_assets_prefix_cell(doc: dict) -> list[dict]:
    """Match tables containing a cell starting with total assets."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            for c in (span.get("table_data") or {}).get("cells", []):
                if (c.get("text") or "").strip().lower().startswith("total assets"):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
