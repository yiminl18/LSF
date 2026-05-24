def rule_state_value_known_jurisdictions(doc: dict) -> list[dict]:
    """Match spans containing common jurisdiction values seen in this filing family."""
    import re
    try:
        pattern = r"\b(Delaware|Washington|New York|Jersey(?:\s*\(Channel Islands\))?)\b"
        out = []
        for span in doc.get("texts", []):
            if re.search(pattern, span.get("text", "") or "", re.I):
                out.append(span)
        return out
    except Exception:
        return []
