def rule_consolidated_balance_sheet_phrase(doc: dict) -> list[dict]:
    """Match spans whose text/path contains consolidated balance sheet(s)."""
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "consolidated balance sheet" in text or "consolidated balance sheet" in path or "consolidated balance sheets" in text or "consolidated balance sheets" in path:
                out.append(span)
        return out
    except Exception:
        return []
