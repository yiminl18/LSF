def rule_tables_with_debt_held_by_public(doc: dict) -> list[dict]:
    """Match tables containing 'Debt held by the public', common in later issues' Federal Debt tables."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r'debt held by the public', text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
