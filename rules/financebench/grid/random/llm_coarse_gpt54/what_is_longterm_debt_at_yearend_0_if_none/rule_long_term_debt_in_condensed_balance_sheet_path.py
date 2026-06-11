def rule_long_term_debt_in_condensed_balance_sheet_path(doc: dict) -> list[dict]:
    """Match tables whose path_text indicates condensed balance sheet and whose text includes debt."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure") or {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if span.get("label") == "table" and re.search(r"condensed.*balance sheet", path, re.I) and re.search(r"\bdebt\b", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
