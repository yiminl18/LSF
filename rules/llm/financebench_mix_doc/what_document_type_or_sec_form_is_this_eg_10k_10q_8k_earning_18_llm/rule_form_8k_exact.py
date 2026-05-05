def rule_form_8k_exact(doc: dict) -> list[dict]:
    """Match exact or near-exact FORM 8-K headings."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.fullmatch(r"\s*FORM\s+8-K\s*", (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
