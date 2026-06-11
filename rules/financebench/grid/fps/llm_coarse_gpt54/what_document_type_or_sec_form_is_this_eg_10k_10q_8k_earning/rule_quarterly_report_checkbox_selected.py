def rule_quarterly_report_checkbox_selected(doc: dict) -> list[dict]:
    """Match selected quarterly-report checkbox/list-item text on page 1, useful for 10-Q filings."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if (
                span.get("page_no") == 1
                and span.get("label") in {"checkbox_selected", "list_item", "text", "section_header"}
                and re.search(r"QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15\(d\)", text, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
