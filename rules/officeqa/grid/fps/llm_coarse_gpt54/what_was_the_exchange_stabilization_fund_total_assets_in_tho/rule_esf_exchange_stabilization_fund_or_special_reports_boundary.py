def rule_esf_exchange_stabilization_fund_or_special_reports_boundary(doc: dict) -> list[dict]:
    """Match tables between Exchange Stabilization Fund and Special Reports headers."""
    import re
    try:
        texts = doc.get("texts", [])
        start = end = None
        for i, s in enumerate(texts):
            if start is None and s.get("label") == "section_header" and re.search(r'EXCHANGE STABILIZATION FUND', s.get("text", ""), re.I):
                start = i
            if start is not None and s.get("label") == "section_header" and re.search(r'SPECIAL REPORTS', s.get("text", ""), re.I):
                end = i
                break
        if start is None:
            return []
        if end is None:
            end = min(len(texts), start + 15)
        return [s for s in texts[start:end] if s.get("label") == "table"]
    except Exception:
        return []
