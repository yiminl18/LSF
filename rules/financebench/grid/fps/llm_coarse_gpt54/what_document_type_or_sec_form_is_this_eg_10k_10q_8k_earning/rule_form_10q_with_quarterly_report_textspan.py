def rule_form_10q_with_quarterly_report_textspan(doc: dict) -> list[dict]:
    """Match FORM 10-Q spans whose text_span mentions quarterly report language."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            text_span = (span.get("text_span") or "").strip()
            if (
                span.get("page_no") == 1
                and re.fullmatch(r"FORM\s+10-Q", text, re.I)
                and re.search(r"QUARTERLY REPORT", text_span, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
