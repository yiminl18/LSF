def rule_tables_with_total_assets_and_balance_sheet_path(doc: dict) -> list[dict]:
    """Match tables whose structure path_text itself contains balance-sheet language."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "balance sheet" in path or "statement of financial position" in path:
                out.append(span)
        return out
    except Exception:
        return []
