def rule_any_span_with_both_state_value_and_ein_value(doc: dict) -> list[dict]:
    """Match spans containing both a jurisdiction value and an EIN-like number."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if re.search(r"\b(Delaware|Washington|New York|Jersey(?:\s*\(Channel Islands\))?)\b", text, re.I) and re.search(r"\b\d{2}-\d{7}\b", text):
                out.append(span)
        return out
    except Exception:
        return []
