def rule_page1_period_line_without_transition(doc: dict) -> list[dict]:
    """Match page-1 period lines and avoid generic transition-period placeholders when possible."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower().strip()
            if span.get("page_no") == 1:
                if re.search(r'for the (fiscal year|quarterly period) ended', txt):
                    out.append(span)
                elif "date of report (date of earliest event reported)" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
