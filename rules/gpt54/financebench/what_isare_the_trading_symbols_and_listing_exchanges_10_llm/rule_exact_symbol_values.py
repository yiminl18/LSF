def rule_exact_symbol_values(doc: dict) -> list[dict]:
    """Match standalone ticker spans commonly used as trading symbols."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.fullmatch(r"[A-Z]{2,6}(?:\d+[A-Z]{0,3})?", txt):
                out.append(span)
        return out
    except Exception:
        return []
