def rule_recent_reading_michigan_reuters(doc: dict) -> list[dict]:
    """Match spans mentioning the most recent/latest reading together with Michigan/Reuters consumer sentiment."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            if re.search(r"(most recent|latest|recent reading|final reading|preliminary reading)", text, re.I) and re.search(r"(university of michigan|michigan/reuters|reuters consumer sentiment|consumer sentiment)", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
