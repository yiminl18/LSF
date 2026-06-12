def rule_tables_with_held_by_the_public(doc: dict) -> list[dict]:
    """Match tables containing 'Held by the public', often adjacent to the gross debt figure."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r'held by the public', text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
