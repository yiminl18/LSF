def rule_page1_period_with_checkbox_context(doc: dict) -> list[dict]:
    """Match spans where the period appears near selected/unselected report-type checkboxes."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            label = span.get("label") or ""
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and label in {"checkbox_selected", "checkbox_unselected", "text", "section_header", "list_item"}:
                if re.search(r'(fiscal year ended|quarterly period ended)', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
