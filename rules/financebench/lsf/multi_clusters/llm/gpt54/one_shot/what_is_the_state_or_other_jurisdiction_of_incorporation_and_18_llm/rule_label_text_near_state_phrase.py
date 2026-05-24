def rule_label_text_near_state_phrase(doc: dict) -> list[dict]:
    """Match text spans whose text itself contains the incorporation/jurisdiction phrase."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if re.search(r"state or other jurisdiction of incorporation|state or other jurisdiction of incorporation or organization", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
