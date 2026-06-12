def rule_most_recent_quarter_prose(doc: dict) -> list[dict]:
    """Match prose spans that mention first/second/third/fourth quarter and a GDP annual rate."""
    import re
    try:
        out = []
        quarter_pat = r"(first|second|third|fourth)\s+quarter"
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(quarter_pat, txt, re.I) and re.search(r"real\s+GDP|gross\s+domestic\s+product|GDP", txt, re.I) and re.search(r"\d+\.\d+\s*percent", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
