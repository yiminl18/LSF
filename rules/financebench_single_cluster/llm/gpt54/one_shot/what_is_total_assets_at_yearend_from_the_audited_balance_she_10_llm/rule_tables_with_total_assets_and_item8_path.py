def rule_tables_with_total_assets_and_item8_path(doc: dict) -> list[dict]:
    """Match total-assets tables whose path_text includes Item 8."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "item 8" in path and "total assets" in (span.get("text") or "").lower():
                out.append(span)
        return out
    except Exception:
        return []
