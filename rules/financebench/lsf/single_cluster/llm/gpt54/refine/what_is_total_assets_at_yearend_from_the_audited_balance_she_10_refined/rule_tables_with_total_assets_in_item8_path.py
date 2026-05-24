def rule_tables_with_total_assets_in_item8_path(doc: dict) -> list[dict]:
    """Match tables containing total assets under Item 8 path."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            low = (span.get("text") or "").lower()
            if "total assets" in low and ("item 8" in path or "financial statements and supplementary data" in path):
                out.append(span)
        return out
    except Exception:
        return []
