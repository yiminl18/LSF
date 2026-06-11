def rule_form_8k_with_current_report_textspan(doc: dict) -> list[dict]:
    """Match FORM 8-K spans whose text_span mentions current report language."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            text_span = (span.get("text_span") or "").strip()
            if (
                span.get("page_no") == 1
                and re.fullmatch(r"FORM\s+8-K", text, re.I)
                and re.search(r"CURRENT REPORT", text_span, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
