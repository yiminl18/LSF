def rule_page2_repeated_form_period(doc: dict) -> list[dict]:
    """Match repeated reporting-period lines on page 2 such as 'AMAZON.COM, INC. FORM 10-K For the Fiscal Year Ended ...'."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 2 and re.search(r'FORM\s+10-(K|Q)', txt, re.I):
                if re.search(r'(Fiscal Year Ended|quarterly period ended)', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
