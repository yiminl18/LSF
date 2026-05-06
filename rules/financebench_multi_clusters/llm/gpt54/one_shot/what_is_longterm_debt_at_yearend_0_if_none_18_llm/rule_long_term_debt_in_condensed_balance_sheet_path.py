def rule_long_term_debt_in_condensed_balance_sheet_path(doc: dict) -> list[dict]:
    """Match spans under paths that include condensed consolidated financial statements and debt."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if "condensed consolidated financial statements" in path and re.search(r"\bdebt\b|\blong[- ]term debt\b", txt):
                out.append(span)
    except Exception:
        return []
    return out
