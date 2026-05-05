def rule_current_and_long_term_debt_text_pair(doc: dict) -> list[dict]:
    """Match text spans that mention both current and long-term debt in one span."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            if (
                ("current portion" in txt or "current maturities" in txt)
                and re.search(r"\blong[\-\s]?term debt\b", txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
