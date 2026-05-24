def rule_current_and_long_term_debt_table(doc: dict) -> list[dict]:
    """Match tables containing both current debt and long-term debt rows."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "").lower()
            if re.search(r"current portion of long[- ]term debt|short[- ]term debt|debt", text) and re.search(r"long[- ]term debt", text):
                out.append(span)
    except Exception:
        return []
    return out
