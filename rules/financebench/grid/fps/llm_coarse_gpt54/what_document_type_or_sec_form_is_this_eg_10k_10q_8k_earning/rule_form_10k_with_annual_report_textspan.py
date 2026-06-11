def rule_form_10k_with_annual_report_textspan(doc: dict) -> list[dict]:
    """Match FORM 10-K spans whose text_span mentions annual report language."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            text_span = (span.get("text_span") or "").strip()
            if (
                span.get("page_no") == 1
                and re.fullmatch(r"FORM\s+10-K", text, re.I)
                and re.search(r"ANNUAL REPORT", text_span, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
