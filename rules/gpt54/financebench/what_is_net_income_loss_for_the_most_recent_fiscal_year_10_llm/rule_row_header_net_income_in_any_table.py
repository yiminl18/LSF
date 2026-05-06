def rule_row_header_net_income_in_any_table(doc: dict) -> list[dict]:
    """Match any table where a row header cell is net income/loss/earnings."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            for c in cells:
                if c.get("is_row_header") and re.search(r"\bnet income\b|\bnet earnings\b|\bnet loss\b|\bnet income \(loss\)\b", c.get("text", "") or "", re.I):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
