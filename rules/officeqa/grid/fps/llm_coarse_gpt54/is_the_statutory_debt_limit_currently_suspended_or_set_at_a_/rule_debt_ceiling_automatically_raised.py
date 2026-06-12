def rule_debt_ceiling_automatically_raised(doc: dict) -> list[dict]:
    """Match spans saying the debt ceiling will be automatically raised after a suspension period."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            if re.search(r"automatically raised", text, re.I) and re.search(r"debt ceiling|debt limit|interim borrowing", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
