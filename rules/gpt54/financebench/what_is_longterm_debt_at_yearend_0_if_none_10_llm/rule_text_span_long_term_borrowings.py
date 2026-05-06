def rule_text_span_long_term_borrowings(doc: dict) -> list[dict]:
    """Match spans using long-term borrowings as the debt label."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            if re.search(r"\blong[\-\s]?term borrowings\b", txt):
                out.append(span)
        return out
    except Exception:
        return []
