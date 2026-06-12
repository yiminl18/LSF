def rule_esf_section_table(doc: dict) -> list[dict]:
    """Match table spans whose text mentions Exchange Stabilization Fund balance sheet / ESF-1."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if (
                re.search(r'\bESF-?1\b', text, re.I)
                or re.search(r'Exchange Stabilization Fund', text, re.I)
                or re.search(r'Balance sheet', text, re.I) and re.search(r'ESF', path + " " + text, re.I)
                or re.search(r'EXCHANGE STABILIZATION FUND', path, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
