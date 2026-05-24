def rule_page1_period_in_any_label(doc: dict) -> list[dict]:
    """Broad high-recall rule for any page-1 span mentioning the reporting period."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if re.search(r'(fiscal year ended|quarterly period ended|date of report \(date of earliest event reported\)|annual report on form 10-k for the fiscal year ended)', txt):
                out.append(span)
        return out
    except Exception:
        return []
