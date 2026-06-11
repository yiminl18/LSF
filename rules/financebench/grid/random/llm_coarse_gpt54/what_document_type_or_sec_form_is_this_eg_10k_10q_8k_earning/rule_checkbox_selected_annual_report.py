def rule_checkbox_selected_annual_report(doc: dict) -> list[dict]:
    """Match selected-checkbox/list-item/text spans indicating ANNUAL REPORT on the cover page."""
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").upper()
            if span.get("page_no") == 1 and "ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(D)" in text:
                if span.get("label") in {"checkbox_selected", "list_item", "text", "section_header"}:
                    out.append(span)
        return out
    except Exception:
        return []
