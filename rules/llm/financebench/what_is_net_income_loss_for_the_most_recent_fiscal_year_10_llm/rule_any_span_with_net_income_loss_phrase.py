def rule_any_span_with_net_income_loss_phrase(doc: dict) -> list[dict]:
    """Match any span explicitly containing the phrase net income (loss) or similar."""
    import re
    try:
        return [
            s for s in doc.get("texts", [])
            if re.search(r"net income\s*\(loss\)|net earnings\s*\(loss\)|net income|net earnings|net loss", s.get("text", "") or "", re.I)
        ]
    except Exception:
        return []
