def rule_esf_contents_entry(doc: dict) -> list[dict]:
    """Match contents-page spans that mention ESF-1 or Exchange Stabilization Fund."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = span.get("text") or ""
            if re.search(r'\bESF-?1\b', text, re.I) or re.search(r'EXCHANGE STABILIZATION FUND', text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
