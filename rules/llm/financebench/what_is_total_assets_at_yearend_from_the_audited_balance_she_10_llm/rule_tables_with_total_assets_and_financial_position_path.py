def rule_tables_with_total_assets_and_financial_position_path(doc: dict) -> list[dict]:
    """Match tables whose path_text contains financial-position language and assets."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if "financial position" in path and "assets" in txt:
                out.append(span)
        return out
    except Exception:
        return []
