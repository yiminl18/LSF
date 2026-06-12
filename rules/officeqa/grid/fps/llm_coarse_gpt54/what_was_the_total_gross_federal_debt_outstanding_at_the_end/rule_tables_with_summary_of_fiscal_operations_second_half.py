def rule_tables_with_summary_of_fiscal_operations_second_half(doc: dict) -> list[dict]:
    """Match the second half/continued FFO-1 table where selected balances and debt totals often appear."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            if span.get("label") != "table":
                continue
            txt = span.get("text", "") or ""
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if re.search(r'selected balances end of period', txt, re.I) and re.search(r'summary of fiscal operations|ffo[\-–— ]?1', path, re.I):
                out.append(span)
    except Exception:
        return []
    return out
