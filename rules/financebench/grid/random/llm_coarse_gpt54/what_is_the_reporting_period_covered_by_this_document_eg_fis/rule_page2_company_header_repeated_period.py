def rule_page2_company_header_repeated_period(doc: dict) -> list[dict]:
    """Match page-2 repeated company/form headers that restate the period."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 2 and re.search(r'(FORM\s+10-K|FORM\s+10-Q)', txt, re.I):
                if re.search(r'(Fiscal Year Ended|quarterly period ended)', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
