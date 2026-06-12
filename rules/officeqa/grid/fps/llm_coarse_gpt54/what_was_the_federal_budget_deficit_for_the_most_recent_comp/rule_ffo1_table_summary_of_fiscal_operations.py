def rule_ffo1_table_summary_of_fiscal_operations(doc: dict) -> list[dict]:
    """Match Table FFO-1 / Summary of Fiscal Operations tables, which may contain deficit rows/columns."""
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if span.get("label") == "table" and (
                "summary of fiscal operations" in txt
                or "table ffo-1" in txt
                or "summary of fiscal operations" in path
            ):
                out.append(span)
    except Exception:
        return []
    return out
