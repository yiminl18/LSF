def rule_status_and_application_of_statutory_limitation_anywhere(doc: dict) -> list[dict]:
    """Match any span containing status and application of statutory limitation."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            if re.search(r"status and application of statutory limitation", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
