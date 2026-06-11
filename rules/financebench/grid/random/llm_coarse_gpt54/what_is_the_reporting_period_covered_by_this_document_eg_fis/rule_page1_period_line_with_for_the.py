def rule_page1_period_line_with_for_the(doc: dict) -> list[dict]:
    """Match page-1 spans beginning with 'For the' and containing ended/reporting-period language."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r'^\s*For the\b', txt, re.I):
                if re.search(r'(fiscal year ended|quarterly period ended|transition period)', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
