def rule_statutory_limit_keyword_anywhere(doc: dict) -> list[dict]:
    """Match any span mentioning statutory limit or statutory limitation."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            if re.search(r"statutory (limit|limitation)", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
