def rule_balance_sheet_path_followed_by_table(doc: dict) -> list[dict]:
    """Match tables whose structural path already contains Consolidated Balance Sheet."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "balance sheet" in path:
                out.append(span)
        return out
    except Exception:
        return []
