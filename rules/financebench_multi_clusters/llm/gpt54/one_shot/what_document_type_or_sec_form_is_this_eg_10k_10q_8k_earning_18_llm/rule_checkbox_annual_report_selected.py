def rule_checkbox_annual_report_selected(doc: dict) -> list[dict]:
    """Match selected/list/checkbox spans containing ANNUAL REPORT PURSUANT..., a strong 10-K cue."""
    try:
        out = []
        for span in doc.get("texts", []):
            label = span.get("label") or ""
            text = (span.get("text") or "").upper()
            if label in {"checkbox_selected", "list_item", "text", "section_header"} and "ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(D)" in text:
                out.append(span)
        return out
    except Exception:
        return []
