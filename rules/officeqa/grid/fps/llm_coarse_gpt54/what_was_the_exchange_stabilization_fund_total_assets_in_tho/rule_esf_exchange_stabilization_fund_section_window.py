def rule_esf_exchange_stabilization_fund_section_window(doc: dict) -> list[dict]:
    """Match all spans in a small window around the Exchange Stabilization Fund header."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("label") == "section_header" and re.search(r'EXCHANGE STABILIZATION FUND', span.get("text", ""), re.I):
                out.extend(texts[max(0, i - 2): min(len(texts), i + 6)])
        return out
    except Exception:
        return []
