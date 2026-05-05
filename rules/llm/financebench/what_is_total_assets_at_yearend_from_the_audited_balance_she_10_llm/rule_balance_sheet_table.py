def rule_balance_sheet_table(doc: dict) -> list[dict]:
    """Match tables that look like the audited balance sheet / consolidated balance sheet."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if (
                "balance sheet" in text
                or "balance sheet" in path
                or "statement of financial position" in text
                or "statement of financial position" in path
            ):
                out.append(span)
        return out
    except Exception:
        return []
