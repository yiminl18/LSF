def rule_university_of_michigan_exact(doc: dict) -> list[dict]:
    """Match spans containing University of Michigan."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r"\buniversity of michigan\b", (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
