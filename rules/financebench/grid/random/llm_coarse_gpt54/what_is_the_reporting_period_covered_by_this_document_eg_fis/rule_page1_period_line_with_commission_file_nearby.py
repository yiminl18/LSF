def rule_page1_period_line_with_commission_file_nearby(doc: dict) -> list[dict]:
    """Match period lines that appear adjacent to commission file number text in the same span."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(r'Commission File Number|Commission file number|Commission File No\.', txt, re.I):
                if re.search(r'(fiscal year ended|quarterly period ended)', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
