def rule_reuters_exact(doc: dict) -> list[dict]:
    """Match spans containing Reuters in a likely sentiment context."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            if re.search(r"\breuters\b", text, re.I) and re.search(r"(michigan|sentiment|consumer)", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
