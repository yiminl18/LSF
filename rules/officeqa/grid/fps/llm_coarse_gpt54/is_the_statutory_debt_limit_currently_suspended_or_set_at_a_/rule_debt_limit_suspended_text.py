def rule_debt_limit_suspended_text(doc: dict) -> list[dict]:
    """Match spans explicitly stating the debt limit was suspended."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            if re.search(r"debt limit (was|is)?\s*suspended", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
