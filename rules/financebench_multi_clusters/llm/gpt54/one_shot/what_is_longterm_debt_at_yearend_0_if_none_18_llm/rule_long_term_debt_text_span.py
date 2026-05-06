def rule_long_term_debt_text_span(doc: dict) -> list[dict]:
    """Match any span whose text explicitly mentions long-term debt."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            if re.search(r"\blong[- ]term debt\b", txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
