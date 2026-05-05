def rule_page1_company_header_with_period_reference(doc: dict) -> list[dict]:
    """Match company-name headers on page 1 whose text_span references the reporting period."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            txt = (span.get("text_span") or "").lower()
            if re.search(r'(fiscal year ended|quarterly period ended|date of report|annual report on form 10-k for the fiscal year ended)', txt):
                out.append(span)
        return out
    except Exception:
        return []
