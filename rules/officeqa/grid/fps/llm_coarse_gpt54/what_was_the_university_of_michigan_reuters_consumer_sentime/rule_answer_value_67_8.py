def rule_answer_value_67_8(doc: dict) -> list[dict]:
    """Match the exact known answer value 67.8 if it appears directly in a span."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r"\b67\.8\b", (span.get("text") or ""))
        ]
    except Exception:
        return []
