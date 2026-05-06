def rule_reporting_period_10q_header_textspan(doc: dict) -> list[dict]:
    """Retrieve page-1 FORM 10-Q cover section headers whose text_span contains the quarterly period ended phrase."""
    try:
        texts = doc.get("texts", []) or []
        out = []
        for span in texts:
            if not isinstance(span, dict):
                continue
            if span.get("page_no") != 1:
                continue
            if span.get("label") != "section_header":
                continue
            st = span.get("structure") or {}
            path = (st.get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            txt_span = (span.get("text_span") or "").lower()
            if "form 10-q" not in path:
                continue
            if "quarterly period ended" in txt_span or ("quarterly report" in txt and "ended" in txt_span):
                out.append(span)
        return out
    except Exception:
        return []

