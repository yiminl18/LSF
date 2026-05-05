def rule_page1_header_with_commission_file_and_period(doc: dict) -> list[dict]:
    """Match page-1 headers that mention both a period phrase and commission file number."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (((span.get("text") or "") + " " + (span.get("text_span") or "")).lower())
            if span.get("page_no") == 1 and "commission file" in txt:
                if re.search(r'(fiscal year ended|quarterly period ended|date of report)', txt):
                    out.append(span)
        return out
    except Exception:
        return []
