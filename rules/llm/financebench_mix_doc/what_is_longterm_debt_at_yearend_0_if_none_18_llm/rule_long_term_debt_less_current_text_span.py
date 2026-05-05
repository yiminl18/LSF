def rule_long_term_debt_less_current_text_span(doc: dict) -> list[dict]:
    """Match any span whose text mentions long-term debt less current portion."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            if re.search(r"long[- ]term debt.*less current portion", txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
