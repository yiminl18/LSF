def rule_michigan_reuters_slash_form(doc: dict) -> list[dict]:
    """Match spans containing Michigan/Reuters slash-style naming."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r"michigan\s*/\s*reuters", (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
