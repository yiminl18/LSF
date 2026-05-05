def rule_tables_with_total_assets_and_balance_keywords_in_path(doc: dict) -> list[dict]:
    """Match tables with total assets whose path contains balance-sheet-related keywords."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            low = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "total assets" in low and (
                "balance sheet" in path
                or "financial position" in path
                or "item 8" in path
            ):
                out.append(span)
        return out
    except Exception:
        return []
