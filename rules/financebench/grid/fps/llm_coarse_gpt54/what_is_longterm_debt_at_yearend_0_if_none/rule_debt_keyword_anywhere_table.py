def rule_debt_keyword_anywhere_table(doc: dict) -> list[dict]:
    """Match any table whose markdown text contains debt-related keywords."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "")
                if re.search(r"\blong[- ]term debt\b|\bdebt\b|\bborrowings\b|\bnotes payable\b", txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
