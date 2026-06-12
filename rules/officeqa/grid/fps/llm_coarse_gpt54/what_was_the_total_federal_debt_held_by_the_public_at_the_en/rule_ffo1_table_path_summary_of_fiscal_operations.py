def rule_ffo1_table_path_summary_of_fiscal_operations(doc: dict) -> list[dict]:
    """Match table spans under the FFO-1 / Summary of Fiscal Operations path."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if (
                ("ffo-1" in path or "table ffo-1" in path or "table ff0-1" in path)
                and ("summary of fiscal operations" in path or "summary of fiscal operations" in txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
