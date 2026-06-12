def rule_debt_ceiling_suspended_text(doc: dict) -> list[dict]:
    """Match spans explicitly stating the debt ceiling was suspended."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            if re.search(r"debt ceiling (was|is)?\s*suspended", text, re.I) or re.search(r"suspended until", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
