def rule_page1_bold_small_period_lines(doc: dict) -> list[dict]:
    """Match likely period lines on page 1 that are bold and contain ended/report dates."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and span.get("bold") == 1 and (span.get("size") or 0) <= 11.5:
                if re.search(r'(ended\s+[A-Z][a-z]+|\bDate of Report\b|\bevent reported\b)', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
