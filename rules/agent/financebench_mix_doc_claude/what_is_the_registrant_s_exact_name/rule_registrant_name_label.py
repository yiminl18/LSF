def rule_registrant_name_label(doc: dict) -> list[dict]:
    """Return span immediately before 'exact name of registrant' label on page 1."""
    try:
        texts = doc.get("texts", [])
        for i, s in enumerate(texts):
            text_lower = s.get("text", "").lower()
            if "exact name" in text_lower and "registrant" in text_lower:
                if i > 0:
                    return [texts[i - 1]]
        return []
    except Exception:
        return []
