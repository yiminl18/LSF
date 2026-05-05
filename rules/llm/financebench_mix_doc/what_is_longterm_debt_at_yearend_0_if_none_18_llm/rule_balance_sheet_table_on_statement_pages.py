def rule_balance_sheet_table_on_statement_pages(doc: dict) -> list[dict]:
    """Match tables on pages whose path indicates financial statements and that likely include debt."""
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if "financial statements" in path and ("balance sheet" in txt or "liabilities" in txt or "debt" in txt):
                out.append(span)
    except Exception:
        return []
    return out
