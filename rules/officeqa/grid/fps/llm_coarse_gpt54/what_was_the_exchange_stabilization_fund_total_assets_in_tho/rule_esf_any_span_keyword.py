def rule_esf_any_span_keyword(doc: dict) -> list[dict]:
    """Match any span mentioning Exchange Stabilization Fund, ESF-1, or total assets."""
    import re
    try:
        return [
            s for s in doc.get("texts", [])
            if re.search(r'Exchange Stabilization Fund|\bESF-?1\b|total assets', s.get("text", "") or "", re.I)
        ]
    except Exception:
        return []
