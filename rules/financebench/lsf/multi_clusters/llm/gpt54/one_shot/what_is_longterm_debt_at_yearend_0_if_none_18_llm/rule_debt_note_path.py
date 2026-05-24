def rule_debt_note_path(doc: dict) -> list[dict]:
    """Match spans under note paths mentioning debt or borrowings."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if re.search(r"\b(note\s+\d+.*)?(long[- ]term debt|debt and credit facilities|borrowings|debt)\b", path):
                out.append(span)
    except Exception:
        return []
    return out
