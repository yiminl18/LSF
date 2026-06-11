def rule_long_term_debt_row_header_cell(doc: dict) -> list[dict]:
    """Return row-header cells themselves when they contain long-term debt labels."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            for c in ((span.get("table_data") or {}).get("cells") or []):
                if c.get("is_row_header") and re.search(r"\blong[\-\s]?term debt\b", c.get("text", "") or "", re.I):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
