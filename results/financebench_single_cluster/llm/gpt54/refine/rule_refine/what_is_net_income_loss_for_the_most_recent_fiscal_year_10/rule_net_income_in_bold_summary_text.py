def rule_net_income_in_bold_summary_text(doc: dict) -> list[dict]:
    """Match bold text spans mentioning net income/earnings/loss, often in summary sections."""
    import re
    try:
        return [
            s for s in doc.get("texts", [])
            if s.get("label") in {"text", "section_header"}
            and s.get("bold") == 1
            and re.search(r"\bnet income\b|\bnet earnings\b|\bnet loss\b", s.get("text", "") or "", re.I)
        ]
    except Exception:
        return []
