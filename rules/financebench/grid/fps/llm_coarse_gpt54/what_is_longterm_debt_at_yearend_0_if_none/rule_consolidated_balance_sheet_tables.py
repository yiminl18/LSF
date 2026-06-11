def rule_consolidated_balance_sheet_tables(doc: dict) -> list[dict]:
    """Match consolidated balance sheet tables, where long-term debt is commonly reported."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "").lower()
            text = (span.get("text") or "").lower()
            if "balance sheet" in path or "balance sheets" in path:
                out.append(span)
            elif "consolidated balance sheet" in text or "consolidated balance sheets" in text:
                out.append(span)
        return out
    except Exception:
        return []
