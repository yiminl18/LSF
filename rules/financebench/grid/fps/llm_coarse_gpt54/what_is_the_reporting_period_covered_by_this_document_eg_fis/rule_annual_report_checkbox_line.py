def rule_annual_report_checkbox_line(doc: dict) -> list[dict]:
    """Match selected annual-report checkbox/list lines on page 1 that often sit next to the answer."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and any(lbl in str(span.get("label")) for lbl in ["checkbox_selected", "list_item", "text"])
            and "annual report pursuant to section 13 or 15(d)" in (span.get("text") or "").lower()
        ]
    except Exception:
        return []
