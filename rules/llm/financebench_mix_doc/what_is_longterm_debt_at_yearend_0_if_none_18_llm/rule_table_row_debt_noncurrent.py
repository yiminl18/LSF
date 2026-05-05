def rule_table_row_debt_noncurrent(doc: dict) -> list[dict]:
    """Match tables with non-current debt style row labels."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            for c in cells:
                t = (c.get("text") or "").lower()
                if re.search(r"(non[- ]current|long[- ]term).{0,20}\bdebt\b", t):
                    out.append(span)
                    break
    except Exception:
        return []
    return out
