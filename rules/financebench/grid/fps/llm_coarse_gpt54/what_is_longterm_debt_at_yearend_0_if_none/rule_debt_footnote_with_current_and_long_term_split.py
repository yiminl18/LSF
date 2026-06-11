def rule_debt_footnote_with_current_and_long_term_split(doc: dict) -> list[dict]:
    """Match debt footnote tables that split current and long-term portions."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if "current portion" in txt and ("long-term debt" in txt or "debt" in txt):
                out.append(span)
        return out
    except Exception:
        return []
