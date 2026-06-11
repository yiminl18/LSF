def rule_long_term_debt_or_none_text(doc: dict) -> list[dict]:
    """Match explicit text spans saying long-term debt is none or absent."""
    try:
        import re
        return [
            span for span in doc.get("texts", [])
            if re.search(r"long[\-\s]?term debt", (span.get("text", "") or ""), re.I)
            and re.search(r"\bnone\b|\bno\b", (span.get("text", "") or ""), re.I)
        ]
    except Exception:
        return []
