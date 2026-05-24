def rule_long_term_debt_numeric_text(doc: dict) -> list[dict]:
    """Match text spans that contain long-term debt and a nearby numeric amount."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            if re.search(r"\blong[- ]term debt\b", txt, re.I) and re.search(r"\$?\s*\(?\d[\d,]*(\.\d+)?\)?", txt):
                out.append(span)
    except Exception:
        return []
    return out
