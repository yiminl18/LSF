def rule_long_term_debt_row_header_cells(doc: dict) -> list[dict]:
    """Match tables containing a row-header cell labeled long-term debt."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            for c in cells:
                txt = (c.get("text") or "").lower()
                if c.get("is_row_header") and re.search(r"\blong[\-\s]?term debt\b", txt):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
