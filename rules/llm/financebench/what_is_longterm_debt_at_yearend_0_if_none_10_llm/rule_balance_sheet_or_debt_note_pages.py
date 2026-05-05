def rule_balance_sheet_or_debt_note_pages(doc: dict) -> list[dict]:
    """Match tables on pages whose path or text suggests either balance sheet or debt note context."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if (
                "balance sheet" in path
                or "balance sheet" in txt
                or "notes to consolidated financial statements" in path and ("debt" in path or "debt" in txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
