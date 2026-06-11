def rule_path_text_is_form(doc: dict) -> list[dict]:
    """Match spans whose structure.path_text itself contains FORM 10-K/10-Q/8-K."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", ((span.get("structure") or {}).get("path_text") or ""), re.I)
        ]
    except Exception:
        return []
