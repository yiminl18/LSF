def rule_long_term_debt_any_cell(doc: dict) -> list[dict]:
    """Match tables containing any cell with long-term debt wording."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            if any(re.search(r"\blong[\-\s]?term debt\b", (c.get("text") or "").lower()) for c in cells):
                out.append(span)
        return out
    except Exception:
        return []
