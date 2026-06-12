def rule_exact_known_quarter_style_lowercase(doc: dict) -> list[dict]:
    """Match lowercase quarter/fiscal style seen in some later answers like 'third quarter, fiscal 1989'."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(r"^(first|second|third|fourth)\s+quarter,?\s+fiscal\s+\d{4}$", re.I)
        return [s for s in texts if pat.match((s.get("text") or "").strip())]
    except Exception:
        return []
