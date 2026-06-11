def rule_page1_text_only_period_lines(doc: dict) -> list[dict]:
    """Match page-1 body text spans that directly state the reporting period."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if (span.get("label") or "") not in {"text", "list_item", "checkbox_selected", "checkbox_unselected"}:
                continue
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'\b(fiscal year|quarterly period)\s+ended\b', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
