def rule_long_term_debt_row_header_cell(doc: dict) -> list[dict]:
    """Match tables where a row-header cell contains long-term debt wording."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            for c in ((span.get("table_data") or {}).get("cells") or []):
                if c.get("is_row_header") and re.search(r"long[- ]term debt|debt.*less current portion", c.get("text") or "", re.I):
                    out.append(span)
                    break
    except Exception:
        return []
    return out
