def rule_debt_note_header(doc: dict) -> list[dict]:
    """Match note headers specifically about debt or borrowings."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "section_header":
                continue
            txt = (span.get("text") or "").lower()
            if re.search(r"\b(note\s+\d+.*)?(long[- ]term debt|debt and credit facilities|debt|borrowings)\b", txt):
                out.append(span)
    except Exception:
        return []
    return out
