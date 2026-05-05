def rule_page1_current_report_block(doc: dict) -> list[dict]:
    """Match 8-K current-report blocks on page 1 that usually contain the event date."""
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (((span.get("text") or "") + " " + (span.get("text_span") or "")).lower())
            if span.get("page_no") == 1 and "current report" in txt and "date of report" in txt:
                out.append(span)
        return out
    except Exception:
        return []
