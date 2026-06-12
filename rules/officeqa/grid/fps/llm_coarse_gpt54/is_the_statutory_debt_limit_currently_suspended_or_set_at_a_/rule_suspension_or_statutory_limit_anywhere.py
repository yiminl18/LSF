def rule_suspension_or_statutory_limit_anywhere(doc: dict) -> list[dict]:
    """Match any span mentioning suspension or statutory debt limit concepts."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            if re.search(r"debt ceiling|debt limit|statutory limitation|statutory limit|suspended until|debt subject to statutory", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
