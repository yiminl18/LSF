def rule_unemployment_rate_chart_caption(doc: dict) -> list[dict]:
    """Match the 'Unemployment Rate' heading/caption in Profile of the Economy."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if "Profile of the Economy" in path and re.search(r"^Unemployment Rate$", text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
