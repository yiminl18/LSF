def rule_page1_form10q_quarterly_report_block(doc: dict) -> list[dict]:
    """Match quarterly-report blocks on page 1 that usually contain the quarter end date."""
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (((span.get("text") or "") + " " + (span.get("text_span") or "")).lower())
            if span.get("page_no") == 1 and "quarterly report pursuant to section 13 or 15(d)" in txt:
                out.append(span)
        return out
    except Exception:
        return []
