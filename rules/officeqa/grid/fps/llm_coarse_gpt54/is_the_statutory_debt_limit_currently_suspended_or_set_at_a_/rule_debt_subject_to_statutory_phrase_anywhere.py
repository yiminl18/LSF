def rule_debt_subject_to_statutory_phrase_anywhere(doc: dict) -> list[dict]:
    """Match any span containing the phrase debt subject to statutory limitation/limit."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            if re.search(r"debt subject to statutory (limit|limitation)", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
