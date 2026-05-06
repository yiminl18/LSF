def rule_tables_with_total_assets_and_financial_statements_path(doc: dict) -> list[dict]:
    """Match total-assets tables whose path_text includes financial statements wording."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "financial statements" in path and "total assets" in (span.get("text") or "").lower():
                out.append(span)
        return out
    except Exception:
        return []
