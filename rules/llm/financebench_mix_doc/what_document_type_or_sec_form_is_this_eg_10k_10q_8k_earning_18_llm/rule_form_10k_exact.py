def rule_form_10k_exact(doc: dict) -> list[dict]:
    """Match exact or near-exact FORM 10-K headings."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.fullmatch(r"\s*FORM\s+10-K\s*", (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
