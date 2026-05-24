def rule_tables_with_total_assets_excluding_executive_tables(doc: dict) -> list[dict]:
    """Match total-assets tables while excluding unrelated personnel/executive tables."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if "total assets" not in txt:
                continue
            if any(bad in txt for bad in ["executive officer", "name | position", "age |", "employee profile"]):
                continue
            out.append(span)
        return out
    except Exception:
        return []
