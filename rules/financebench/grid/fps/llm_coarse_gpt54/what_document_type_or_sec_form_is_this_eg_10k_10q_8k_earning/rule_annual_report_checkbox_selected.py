def rule_annual_report_checkbox_selected(doc: dict) -> list[dict]:
    """Match selected annual-report checkbox/list-item text on page 1, useful for 10-K filings."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if (
                span.get("page_no") == 1
                and span.get("label") in {"checkbox_selected", "list_item", "text", "section_header"}
                and re.search(r"ANNUAL REPORT PURSUANT TO SECTION 13 OR 15\(d\)", text, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
