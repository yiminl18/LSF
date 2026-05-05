def rule_page1_bold_small_period_lines(doc: dict) -> list[dict]:
    """Match likely period lines on page 1 that are bold and relatively small body text."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            size = span.get("size") or 0
            if span.get("page_no") == 1 and span.get("bold") == 1 and 6 <= size <= 11.5:
                if re.search(r'(fiscal year|quarterly period|date of report|earliest event reported)', txt):
                    out.append(span)
        return out
    except Exception:
        return []
