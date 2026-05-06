def rule_page2_repeated_form_period_line(doc: dict) -> list[dict]:
    """Match repeated cover/header period lines on page 2."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if span.get("page_no") == 2 and re.search(r'for the (fiscal year|quarterly period) ended', txt):
                out.append(span)
        return out
    except Exception:
        return []
